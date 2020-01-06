
import path
import os
import pandas as pd
import json
import pathlib
from decorators import mkDir

class IOHandler:

    def __init__(self):

        # Locations
        self.holidayFolder = path.Path("holiday/")
        self.logFolder = path.Path("log/")
        self.cacheFolder = path.Path("cache/")
        self.metaFileLocation = self.cacheFolder.joinpath("meta.json")
        self.databaseFolder = path.Path("database/")

        self.mkPaths = {'holiday': lambda: self.holidayFolder.mkdir() if self.holidayFolder.exists() is False else None,
                        'log': lambda: self.logFolder.mkdir() if self.logFolder.exists() is False else None,
                        'cache': lambda: self.cacheFolder.mkdir() if self.cacheFolder.exists() is False else None,
                        'database': lambda: self.databaseFolder.mkdir() if self.databaseFolder.exists() is False else None}

    def mkDirs(self, key):
        self.mkPaths[key]()

    def loadHolidays(self):
        files = self.holidayFolder.listdir()
        self.holidays = {file: pd.read_excel(self.holidayFolder.joinpath('{0}'.format(file))).dropna() for file in
                         files}

    def getKey(self):
        with open('alpha.key') as file:
            self.keys = file.readlines()

    @staticmethod
    def mkdir(pathStr):
        pathlib.Path(pathStr).mkdir(parents=True, exist_ok=True)

    @staticmethod
    @mkDir
    def writeJSON(obj, pathStr, fileName=''):
        json.dump(obj, open(path.Path(pathStr).joinpath(fileName) if len(fileName) > 0 else pathStr, "w"),
                  indent=4)

    @staticmethod
    def readJSON(pathStr, fileName=''):
        return json.load(open(path.Path(pathStr).joinpath(fileName) if len(fileName) > 0 else pathStr, "r"))

    @staticmethod
    @mkDir
    def df2Excel(obj, pathStr, fileName=''):
        obj.to_excel(path.Path(pathStr).joinpath(fileName) if len(fileName) > 0 else pathStr)

    @staticmethod
    @mkDir
    def df2parquet(obj, pathStr, fileName=''):

        pathStr = path.Path(pathStr).joinpath(fileName) if len(fileName) > 0 else pathStr
        obj.to_parquet(pathStr,
                       compression='snappy',
                       engine='fastparquet')

    @staticmethod
    def readParquet(pathStr, fileName=''):
        return pd.read_parquet(path.Path(pathStr).joinpath(fileName) if len(fileName) > 0 else pathStr,
                               engine='fastparquet')


