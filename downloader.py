#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 16:38:49 2019

@author: joaopedroaugusto
"""

import datetime as dt
import time 
import requests
import json
import pandas as pd
import multiprocessing as mp
from io_handler import IOHandler
import sys
import CONSTANTS
import utility
import dateparser
import random

FORCE_DOWNLOAD = False

class AlphaDownloader(IOHandler):

    # Always download data from the beggining and keep up to date, there is no
    # start-end dates here as we want to build a full database
    # Data validation to be done in a later process for pqt file creation, this is just the downloader and updater
    def __init__(self, processes=10):

        super().__init__()
        self.mkDirs('holiday')
        self.mkDirs('log')
        self.mkDirs('cache')

        self.functions = ['TIME_SERIES_DAILY','TIME_SERIES_DAILY_ADJUSTED','TIME_SERIES_INTRADAY','FX_DAILY','DIGITAL_CURRENCY_DAILY']
        self.indicators = ['.SA']
        self.downloadGuidelines = {function: {} for function in self.functions}
        self.failedDownloads = {function: {} for function in self.functions}
        
        self.data = {function: {} for function in self.functions}
        self.interval = None

        # Reading alpha key
        self.getKey()
        
        # Download control variables
        self.processes=processes

        # 5 calls per minute + safety factor of 5 seconds
        self.batchWaitTime = 60 + 5
        self.callsPerMinute = 30
        self.batchFinished = False
        self.batchCount = 0
        
        # Meta
        self.metaData = {function: {} for function in self.functions}
        self.loadMeta()
        self.loadHolidays()

        # log
        self.log = {}
    
    def checkHoliday(self, date, indicator):
        
        if indicator == '.SA':
            return (date in self.holidays['brazil.xls']['Data'].apply(lambda dt: dt.date()).values)
        else:
            raise Exception("Indicator {0} not implemented ... yet".format(indicator))
            
    
    # Read meta file to update db
    def loadMeta(self):

        if self.metaFileLocation.exists():
            existingMeta = json.load(open(self.metaFileLocation, 'r'))
            for function in existingMeta.keys():
                if function not in self.metaData.keys():
                    self.metaData[function] = {}
                else:
                    self.metaData[function].update(existingMeta[function])

    # Register ticket and update guidelines for the download
    # Extra arguments should be added for other downloads, like interval=5min for intraday
    # Check API documentation
    def registerTicker(self, function, ticker, extraArguments={}):
        
        if function not in self.functions:
            raise Exception("Not implemented {0} function ... yet".format(function))

        payload = {}
        # Used for saving in the files
        payload['save_symbol'] = ticker

        # Used for downloading
        payload['function'] = function
        payload['symbol'] = ticker
        payload['datatype'] = 'json'
        payload.update(extraArguments)

        # Update check
        if ticker in self.metaData[function].keys() and not FORCE_DOWNLOAD:

            if function in ['TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED']:

                now = dt.date.today()

                # TODO: Last Refreshed is not a good parameter because it is the date the server was last updated
                # We should create a Last Refreshed for our operational procedure and check according to both
                lastUpdated = self.metaData[function][ticker]['3. Last Refreshed']
                lastUpdatedObj = dateparser.parse(lastUpdated).date()

                # Keep database up to date until last working day
                lastWorkingDay = utility.prev_weekday(now)

                #Ignore sat, sun and holidays, check different timezones?
                indicators_in = [ind in ticker for ind in self.indicators]
                if any(indicators_in):
                    idx = indicators_in.index(True)
                    isHoliday = self.checkHoliday(lastWorkingDay, self.indicators[idx])
                else:
                    isHoliday = False

                if (lastUpdatedObj < lastWorkingDay): #dt.timedelta(days=1) for testing
                    if not isHoliday:
                        # Download compact to update only, faster dl
                        payload['outputsize'] = 'compact'
                    else:
                        print("Holiday. No update for {0} today, fag.".format(ticker))
                        payload['outputsize'] = None
                else:
                    # Dont do anything
                    payload['outputsize'] = None
            
            #TODO: add rules for other function updates!
            else:
                payload['outputsize'] = 'full'
        else:
            
            # Adding extra arguments
            payload.update(extraArguments)

            # IF FOREX, SPLIT
            if function in ['FX_DAILY', 'DIGITAL_CURRENCY_DAILY']:

                from_ = ticker[0:3]
                to_ = ticker[3:]

                if function == 'FX_DAILY':
                    payload['from_symbol'] = from_
                    payload['to_symbol'] = to_

                elif function == 'DIGITAL_CURRENCY_DAILY':
                    payload['symbol'] = from_
                    payload['market'] = to_

                # Future implementation
                else:
                    pass

            elif function in ['TIME_SERIES_INTRADAY']:
                if 'interval' not in payload.keys():
                    raise Exception("interval extra argument required for intraday download.")

            # Download full data
            payload['outputsize'] = 'full'

        self.downloadGuidelines[function][ticker] = payload
        self.log[ticker] = {'function': function,
                            'status': 'registered',
                            'when': dt.datetime.now().strftime(CONSTANTS.dateFormat),
                            'date': None,
                            'outputsize': payload['outputsize']}

        self.writeJSON(self.log, self.logFolder, '{0}.json'.format(dt.datetime.now().strftime(CONSTANTS.dateFormat)))

    def makeLogXlsx(self, file):
        file = file.replace(".json", "")
        fileJson = "{0}.json".format(file)
        fileXlsx = "{0}.xlsx".format(file)

        dict_ = self.readJSON(self.logFolder, fileJson)
        self.df2Excel(pd.DataFrame(dict_).T, self.logFolder, fileXlsx)
    
    # Static method for parallelization with multiprocessing
    @staticmethod
    def downloadAlphaData(payload):
        
        endpoint = "https://www.alphavantage.co/query"
        result = requests.get(endpoint, payload)
        return result
    
    def updateOrCreateData(self, function, ticker, content):

        functionFolder = self.cacheFolder.joinpath(function)
        tickerFileName = '{0}.json'.format(ticker)
        isUpdate = self.downloadGuidelines[function][ticker]['outputsize'] == 'compact'
        contentKey = [key for key in content.keys() if key != 'Meta Data'][0]

        self.metaData[function][ticker] = content['Meta Data']

        ## TODO: CONSERTAR ESSA MERDA HORRIVEL
        for key in content['Meta Data'].keys():
            if 'Last Refreshed' in key:
                refresh_key = key

        # Saving log
        self.log[ticker]['date'] = content['Meta Data'][refresh_key]
        self.log[ticker]['status'] = 'downloaded'
        # Go
        content = content[contentKey]

        if isUpdate:
            existingContent = self.readJSON(functionFolder, tickerFileName)
            existingContent.update(content)
            content = existingContent

        if not functionFolder.exists(): functionFolder.mkdir()

        self.writeJSON(content, functionFolder, tickerFileName)
        self.writeJSON(self.metaData, self.metaFileLocation)
        self.writeJSON(self.log, self.logFolder, '{0}.json'.format(dt.datetime.now().strftime(CONSTANTS.dateFormat)))
    
    # This step should also be parallelizable
    def makeAndCache(self, function, ticker, result):
        
        if result.ok:      
            content = json.loads(result.content)
            if 'Error Message' in content.keys():
                print("Error Message")
                self.failedDownloads[function][ticker] = content
            else:
                if 'Meta Data' in content.keys():
                    self.updateOrCreateData(function, ticker, content)
                else:
                    print("No Meta Data")
                    #self.failedDownloads[function][ticker] = content
                    return True
        else:
            print("Result not ok")
            #self.failedDownloads[function][ticker] = 'Connection Error'
            return True

        self.writeJSON(self.failedDownloads, self.cacheFolder, 'failed.json')
        return False

    @staticmethod
    def getUsableKey(keyDict):
        for key in keyDict.keys():
            if (keyDict[key] is None) or (dt.datetime.now() > keyDict[key]):
                return key
        return None
    
    def downloadData(self):
        
        self.batchFinished = False
        self.batchCount = 0
        
        # Flattening payloads where outputsize is not None
        self.workArgs = [self.downloadGuidelines[f][t] for f in self.downloadGuidelines.keys() for t in self.downloadGuidelines[f] if self.downloadGuidelines[f][t]['outputsize'] != None]

        # Configuring multiprocessing
        usedProcesses = self.processes if self.processes < self.callsPerMinute else 1
        csize = int(max([self.callsPerMinute/usedProcesses,1]))
        print("Starting download with {0} processes and chunk size {1} of {2} downloadGuidelines.".format(usedProcesses, csize, len(self.workArgs)))
        print("Calling {0} times per minute with {1} keys, process should take {2} minutes.".format(self.callsPerMinute, len(self.keys), len(self.workArgs)/(self.callsPerMinute *len(self.keys))))
        
        pool = mp.Pool(usedProcesses)
        keyJuggler = {key: None for key in self.keys}
        start = time.time()
        while not self.batchFinished:

            # Get batch
            batch = self.workArgs[self.batchCount: self.batchCount + self.callsPerMinute]

            # Get usable key and add to batch
            key = self.getUsableKey(keyJuggler)
            if key is not None:
                print("Using key: ", key)
                for gl in batch:
                    gl['apikey'] = key

                # Start download
                self.downloadedResults = pool.map(AlphaDownloader.downloadAlphaData, batch, chunksize=csize)

                # Updating key timer
                keyJuggler[key] = dt.datetime.now() + dt.timedelta(seconds=self.batchWaitTime)

                # Building result
                for data in zip(batch, self.downloadedResults):
                    print("Building ticker: {0} - {1}".format(data[0]['function'], data[0]['save_symbol']))
                    frequencyError = self.makeAndCache(data[0]['function'], data[0]['save_symbol'], data[1])
                    if not frequencyError:
                        self.batchCount += 1

                if frequencyError:
                    print("Too frequent error.")

                print("Ready: {0:.2f}%".format(self.batchCount / len(self.workArgs) * 100))
                time.sleep(random.choice([7,8,9,10,11,12]))
            else:
                print("Awaiting for key.")
                time.sleep(10)

            if self.batchCount >= len(self.workArgs):
                self.batchFinished = True
            else:
                print('Estimated calls/min: ',self.batchCount/((time.time()-start)/60.0))

        pool.close()

if __name__ == '__main__':

    print("Running downloader . . . ")
    dw = AlphaDownloader()
    if len(sys.argv) >= 2:
        settings_location = sys.argv[1]
        print("Settings location: ", settings_location)
        if len(sys.argv) > 2:
            FORCE_DOWNLOAD = eval(sys.argv[2])
    else:
        print("Not sufficient arguments")
        exit()

    settings = json.load(open(settings_location, "r"))
    functions = settings.keys()

    print("[FORCE DL]: ", FORCE_DOWNLOAD)

    for function in functions:
        for ticker in settings[function]:
            extraArgs = settings[function][ticker]
            dw.registerTicker(function, ticker, extraArgs)
    dw.downloadData()