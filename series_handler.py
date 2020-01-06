import datetime as dt
import time
import json
import pandas as pd
import os
import sys
import numpy as np
import utility
import CONSTANTS
import multiprocessing as mp
import tqdm
from io_handler import IOHandler
import sys

GLOBAL_VAR_DICT = {}

# Using a global IOHandler because I dont want to extend every dataframe with IO Methods
IO_HANDLER = IOHandler()

def worker_init(counter):
    global GLOBAL_VAR_DICT
    GLOBAL_VAR_DICT['counter'] = counter

@pd.api.extensions.register_dataframe_accessor("dynamic")
class DynamicDataFrame(object):
    def __init__(self, pandasDataFrameObj):
        self._obj = pandasDataFrameObj
        self.meta = {'created': dt.datetime.now().strftime(CONSTANTS.timeFormat)}
        self.rawDataLink = None
    
    def setMeta(self, meta):
        self.meta = meta
    
    def linkRaw(self, link):
        self.meta['rawDataLink'] = link 
        
    def writeParquet(self, function, ticker):
        
        self.meta['last saved'] = pd.datetime.now().strftime(CONSTANTS.timeFormat)
        self.meta['size'] = sys.getsizeof(self._obj)
        self.meta['columns'] = [name for name in self._obj.columns]

        self._obj.columns = [repr(col) for col in self._obj.columns]

        fileName = '{0}/{1}.pqt'.format(function, ticker)
        fileNameMeta = '{0}/{1}.meta'.format(function, ticker)
        IO_HANDLER.df2parquet(self._obj, IO_HANDLER.databaseFolder, fileName)
        IO_HANDLER.writeJSON(self.meta, IO_HANDLER.databaseFolder, fileNameMeta)
    
    @staticmethod
    def readParquet(function, ticker, cols=None):

        fileName = '{0}/{1}.pqt'.format(function, ticker)
        fileNameMeta = '{0}/{1}.meta'.format(function, ticker)
        try:
            data = IO_HANDLER.readParquet(IO_HANDLER.databaseFolder, fileName)
            columns = [eval(col) for col in data.columns]
            columns = [eval(col) if "(" in col else col for col in columns]
            data.columns = columns
            meta = IO_HANDLER.readJSON(IO_HANDLER.databaseFolder, fileNameMeta)
            data.dynamic.setMeta(meta)
            return data
        except FileNotFoundError:
            return None

@pd.api.extensions.register_dataframe_accessor("financial")
class FinancialMethods(object):
    def __init__(self, pandasDataFrameObj):
        self._obj = pandasDataFrameObj
    
    @staticmethod
    def iterMonad(arg):

        if type(arg) == str:
            arg = [arg]
        else:
            assert(hasattr(arg, '__iter__'))
        return arg
    
    def filterColumns(self, columns):
        columns = list(filter(lambda col: col in self._obj.columns, columns))
        return columns

    def addDates(self, threshold):
        weekday = [ts.to_pydatetime() for ts in self._obj.index]
        prev_weekday = [utility.prev_weekday(wd) for wd in weekday]
        self._obj['prev_wd'] = prev_weekday
        self._obj['shifted_index'] = [pd.NaT]+list(self._obj.index[:-1])
        self._obj['valid_observation'] = (self._obj['prev_wd'] - self._obj['shifted_index']) <= np.timedelta64(threshold,'D').astype('timedelta64[ns]')

    def addReturn(self, overColumn):

        overColumn = FinancialMethods.iterMonad(overColumn)
        overColumn = self.filterColumns(overColumn)

        for column in overColumn:

            maskIsNA = self._obj[column].isna()
            return_ = self._obj[column].pct_change(fill_method='ffill')
            logreturn_ = np.log(1.0+return_)
            self._obj[(column, 'return')] = return_
            self._obj[(column, 'logreturn')] = logreturn_
            self._obj.loc[maskIsNA, (column, 'return')] = np.nan
            self._obj.loc[maskIsNA, (column, 'logreturn')] = np.nan
    
    def addVolatility(self, overColumn, window):

        overColumn = FinancialMethods.iterMonad(overColumn)
        overColumn = self.filterColumns(overColumn)

        for column in overColumn:
            std = self._obj[column].rolling(window, min_periods=0).std()
            self._obj[(column, 'vol daily', window)] = std
            self._obj[(column, 'vol yearly parametric', window)] = std * np.sqrt(252)

    def addDrawdown(self, overColumn, window):

        overColumn = FinancialMethods.iterMonad(overColumn)
        overColumn = self.filterColumns(overColumn)

        for column in overColumn:
            max = self._obj[column].rolling(window, min_periods=0).max()
            dd = self._obj[column]/max - 1
            self._obj[(column, 'drawdown', window)] = dd
    
    @staticmethod
    def profitProbability(serie, inverted=True):
        if inverted:
            buy_value = serie.tail(1).values
        else:
            buy_value = serie.head(1).values
        return_ = (serie.values / buy_value)
        p = (return_ >= 1).sum()/len(return_)
        return p

    def leaveWithProfitProbability(self, overColumn, window):

        overColumn = FinancialMethods.iterMonad(overColumn)
        overColumn = self.filterColumns(overColumn)
        
        for column in overColumn:
            # inverting for forward window
            p = self._obj[column][::-1].rolling(window, min_periods=window).agg(FinancialMethods.profitProbability)
            self._obj[(column, 'profit probability', window)] = p
            
class SeriesHandler(IOHandler):

    def __init__(self, meta=None):

        super().__init__()

        print("Creating series handler at {0}".format(dt.datetime.now().strftime(CONSTANTS.timeFormat)))

        # TODO ler meta data, pegar todas funcoes + tickers, ler jsons e criar em pastas separadas os parquet files
        # Ler json, transformar em dataframe, adicionar coisas como:
        # - retorno, max dd, vol, vol parametrica, alguns indices simples pra comecar
        # calcular tudo e salvar no parquet
        # Filtrar preco zero
        # Conferir datas

        if self.metaFileLocation.exists() and meta is None:
            self.meta = self.readJSON(self.metaFileLocation)
        elif meta is not None:
            self.meta = meta
        else:
            print("Process killed due to missing meta.json file.")
            exit()
        self.financialSeries = {function: {} for function in self.meta.keys()}
        
        # Making path tree
        self.mkDirs('database')
        for function in self.meta.keys():
            if not self.databaseFolder.joinpath(function).exists():
                self.mkdir(self.databaseFolder.joinpath(function))

        self.financialSeriesSize = 0
        # At least one, at most cpu_count - 2
        self.processes = max([int(mp.cpu_count() - 2),1])
        self.pool = None
        print("# of processes: ",self.processes)
    
    @staticmethod
    def doWork(workload):

        global GLOBAL_VAR_DICT

        function, ticker = workload
        link = 'cache/{0}/{1}.json'.format(function, ticker)
        # Read, parse, sort
        frame = pd.DataFrame(json.load(open(link,'r')), dtype=np.float32).T
        print(frame.index)
        index = list(map(lambda strIndex: utility.parseTime(strIndex), frame.index))
        frame.index = index
        frame.sort_index(inplace=True)
        # Fixing columns
        columns = list(map(lambda name: "".join(name.split(" ")[1:]), frame.columns))
        windows = [5, 10, 15, 21, 63, 126, 252]
        frame.columns = columns
        
        # Updating meta
        frame.dynamic.linkRaw(link)

        # Adding previous weekday

        # 3 Days max difference
        frame.financial.addDates(5)

        # Return on all columns
        frame.financial.addReturn(['open', 'close', 'adjustedclose', 'high', 'low', 'volume'])
        
        # 252 window volatility
        #frame.financial.addVolatility(['close return', 'close logreturn', 'volume'], windows[-1])
        
        ## Drawdown
        #[frame.financial.addDrawdown(['close'], w) for w in windows]

        # Profit probability
        #[frame.financial.leaveWithProfitProbability(['close'], w) for w in windows]
        
        # Writing Parquet
        frame.columns = [repr(col) if type(col) is not str else col for col in frame.columns]
        frame.dynamic.writeParquet(function, ticker)
        #print("Finished: ", link)

        if 'counter' in GLOBAL_VAR_DICT.keys():
            with GLOBAL_VAR_DICT['counter'].get_lock():
                GLOBAL_VAR_DICT['counter'].value += 1

    def createDataFramesSingleCore(self):

        workload = list((function, ticker) for function in self.meta.keys() for ticker in self.meta[function].keys())
        total_load = len(workload)
        print('Workload: ',workload)
        [SeriesHandler.doWork(load) for load in workload]
        print("Done")
    
    def createDataFrames(self):
        
        counter = mp.Value('i', 0)
        workload = list((function, ticker) for function in self.meta.keys() for ticker in self.meta[function].keys())
        total_load = len(workload)
        print('Workload: ',workload)
        self.pool = mp.Pool(self.processes,initializer=worker_init, initargs=(counter,))
        self.pool.map_async(SeriesHandler.doWork, workload, chunksize=2)
        # No more work is allowed in the pool
        self.pool.close()
        print("Total load: ",total_load)
        pbar = tqdm.tqdm(total=total_load)

        oldval = 0
        while True:

            time.sleep(0.1)
            read = counter.value
            if read != oldval:
                pbar.update(read - oldval)
                oldval = read

            if counter.value == total_load:
                break

        # Wait all jobs
        self.pool.join()
        print("Ready: {0:.4f}%".format(counter.value/total_load*100))
        print("Counter: {0}".format(counter.value))
if __name__ == '__main__':

    meta = None
    if len(sys.argv) > 1:
        if '.json' in sys.argv[1]:
            file_ = sys.argv[1]
            meta = json.load(open(file_, 'r'))
        else:
            function, ticker = sys.argv[1], sys.argv[2]
            meta = {function: {ticker:{}}}

    #meta = {"TIME_SERIES_DAILY_ADJUSTED": {"NESN.SWI": {}}}
    sh = SeriesHandler(meta)
    #sh.createDataFramesSingleCore()
    sh.createDataFrames()