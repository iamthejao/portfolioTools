import pandas as pd
import numpy as np
from io_handler import IOHandler
from series_handler import DynamicDataFrame, FinancialMethods
from sklearn.linear_model import LinearRegression
import multiprocessing as mp

def loadParquet(args):
    function, ticker = args
    return (args, DynamicDataFrame.readParquet(function, ticker))

class PortfolioMethods(IOHandler):

    def __init__(self):

        super().__init__()
        self.cache = {}
        self.cachedAssets = []
        self.pool = mp.Pool(mp.cpu_count())

    def notInCache(self, guideline):
        function, ticker = guideline
        if function in self.cache.keys():
            if ticker in self.cache[function].keys():
                return False
            else:
                return True
        else:
            return True

    def loadSeries(self, workload):

        if type(workload) is tuple:
            workload = [workload]

        #workload = self.flatDict(workload)
        workload = list(filter(self.notInCache, workload))
        print("Loading size: ", len(workload))

        loaded = self.pool.imap_unordered(loadParquet, workload, chunksize=1)
        for guideline, data in loaded:
            if data is not None:
                function, ticker = guideline
                if function not in self.cache.keys():
                    self.cache[function] = {}
                self.cache[function][ticker] = data
                self.cachedAssets.append(guideline)
            else:
                print("{0} not found.".format(guideline))

    def __getitem__(self, key):
        if type(key) in (list, tuple): return self.cache[key[0]][key[1]]
        else:
            if key in self.cache.keys():
                return self.cache[key]
            else:
                for function in self.cache.keys():
                    if key in self.cache[function]:
                        return self.cache[function][key]

    @staticmethod
    def _flatDict(dict_):
        if type(dict_) is dict:
            return [(function, ticker) for function in dict_.keys() for ticker in dict_[function].keys()]
        else:
            return dict_

    @staticmethod
    def _applyDateFilter(obj, from_, to_):
        if from_ is not None:
            obj = obj[obj.index >= from_]
        if to_ is not None:
            obj = obj[obj.index <= to_]
        return obj

    def getPortfolioDateRange(self, portfolio):
        dateRanges = [(self.cache[function][ticker].index.min(), self.cache[function][ticker].index.max()) for function,ticker in portfolio]
        return min([tpl[0] for tpl in dateRanges]), max([tpl[1] for tpl in dateRanges])

    @staticmethod
    def assertUnique(values):
        assert(len(np.unique(values)) == len(values))

    ## TODO: Other frequencies
    ## TODO: Mask valid only works for 1D
    def getAssetsAttribute(self, assets, on_column, dateRange=(None, None)):

        for guideline in assets:
            if self.notInCache(guideline):
                self.loadSeries(guideline)

        # Adjusting dates
        init, end = self.getPortfolioDateRange(assets)
        init = init if dateRange[0] is None else dateRange[0]
        end = end if dateRange[1] is None else dateRange[1]

        index = pd.date_range(start=init, end=end, freq="1D")
        mergeInto = pd.DataFrame(index=index)
        nameCol = on_column
        for function, ticker in assets:
            data = self.cache[function][ticker]
            mask_valid = data['valid_observation']
            returns = data[nameCol].loc[mask_valid]
            mergeInto = pd.merge(mergeInto, returns, how='left', left_index=True, right_index=True)
        mergeInto.columns = assets
        return_ = mergeInto.dropna(axis=0, how='all')
        self.assertUnique(return_.index)
        return return_

    @staticmethod
    def _pReturn(returns, allocation, direction):
        totalReturns = np.cumprod(1+returns,0) - 1
        contributions = totalReturns * (allocation * direction)
        cumulative = (contributions.sum(1) + 1)
        return cumulative, (1+contributions)

    def getPortfolioReturns(self, assetsReturns, allocation, direction):

        data = assetsReturns.values
        allocation = np.array(allocation)

        cumreturn, cumcontributions = PortfolioMethods._pReturn(data, allocation.astype(np.float32), direction.astype(np.float32))
        portfolioReturns = pd.DataFrame(cumreturn, index=assetsReturns.index, columns=['cumreturn'])
        portfolioReturns['return'] = portfolioReturns['cumreturn'].pct_change().fillna(0.0)
        return portfolioReturns, cumcontributions

    def getRiskFactors(self, portfolioDates, riskFactors):

        if type(riskFactors) not in (list, tuple): riskFactors = [riskFactors]
        mergeInto = pd.DataFrame(index=portfolioDates)
        for factor in riskFactors:
            returns = self[factor][('close', 'return')]
            mergeInto = pd.merge(mergeInto, returns, how='left', left_index=True, right_index=True)
        riskFactorReturns = mergeInto.fillna(0.0)
        riskFactorReturns.columns = riskFactors
        self.assertUnique(riskFactorsReturns.index)
        return riskFactorReturns

    ## TODO: Get column using meta saved on Rf
    ## TODO: have in rf meta how I should treat the values, for now accepting only n/360
    def getRiskFree(self, portfolioDates, rf):
        if type(rf) not in (pd.Series, pd.DataFrame): rf = self[rf][[('4 WEEKS', 'COUPON EQUIVALENT')]]/360
        mergeInto = pd.DataFrame(index=portfolioDates)
        rf = pd.merge(mergeInto, rf, how='left', left_index=True, right_index=True)
        self.assertUnique(rf.index)
        return rf

    @staticmethod
    def _alphaBeta(excessPortfolioReturn, excessRiskFactorsReturn, intercept=True):
        model = LinearRegression(fit_intercept=intercept)
        model.fit(excessRiskFactorsReturn, excessPortfolioReturn)
        alpha = model.intercept_
        beta = model.coef_

        reconstruct = (beta * excessRiskFactorsReturn).sum(1)[None].T + alpha
        error = excessPortfolioReturn - reconstruct

        return alpha, beta, error

    @staticmethod
    def _trackingError(portfolioReturnsVec, trackedIndexReturnsVec):
        return np.std(portfolioReturnsVec - trackedIndexReturnsVec)

    @staticmethod
    def _informationRatio(portfolioReturnsVec, trackedIndexReturnsVec):
        te = np.std(portfolioReturnsVec - trackedIndexReturnsVec)
        ir = (portfolioReturnsVec - trackedIndexReturnsVec).mean()/te
        return ir

    @staticmethod
    def _drawdown(portfolioReturns, window_size=21, top=1, retUnderwater=False):

        portfolioReturns['cumreturn'] = (1+portfolioReturns['return']).cumprod()
        portfolioReturns['max'] = portfolioReturns['cumreturn'].rolling(window=window_size, min_periods=1).max()
        portfolioReturns['dd'] = (portfolioReturns['cumreturn']/portfolioReturns['max']) - 1.0

        if retUnderwater is False:
            dd = portfolioReturns['dd'].sort_values()
            return dd.head(top)
        else:
            return portfolioReturns['dd']


    @staticmethod
    def _drawup(portfolioReturns, window_size=21, top=1, retOverwater=False):

        portfolioReturns['cumreturn'] = (1 + portfolioReturns['return']).cumprod()
        portfolioReturns['min'] = portfolioReturns['cumreturn'].rolling(window=window_size, min_periods=1).min()
        portfolioReturns['du'] = (portfolioReturns['cumreturn'] / portfolioReturns['min']) - 1.0
        dd = portfolioReturns['du'].sort_values(ascending=False)

        if retOverwater is False:
            dd = portfolioReturns['du'].sort_values()
            return dd.head(top)
        else:
            return portfolioReturns['du']





