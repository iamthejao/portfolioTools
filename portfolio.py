import portfolio_methods
import dateparser
import numpy as np
from decorators import timeit
import datetime as dt
import pandas as pd
from seaborn import distplot

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import bytespdate2num, num2date
from matplotlib.ticker import Formatter

plt.style.use(u'grayscale')
font = {'size'   : 50}
matplotlib.rc('font', **font)

PM = portfolio_methods.PortfolioMethods()

class FinancialDateFormatter(Formatter):
    def __init__(self, dates, fmt='%Y-%m-%d'):
        self.dates = dates
        self.fmt = fmt
    def __call__(self, x, pos=0):
        'Return the label for time x at position pos'
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''
        return self.dates[ind].to_pydatetime().strftime(self.fmt)


# Mostly a wrapper for the methods created in portfolio_methods and portfolio_optimizer
class Portfolio:

    def __init__(self, assets, allocation, direction, riskFree, firstPriceOpen=False, renormalize=False):

        # Global PM is responsible for caching all data so every time an asset timeseries is used
        # it is loaded and cached only once and then it is ready to use for every other portfolio class object
        global PM

        self.assets = assets
        self.allocation = np.array(allocation)
        self.direction = np.array(direction)
        self.dateRange = (None, None)
        self.riskFree = ("RISK_FREE", riskFree) if type(riskFree) is str else riskFree
        self.freq = '1D'
        self.factors = [("TIME_SERIES_DAILY_ADJUSTED", "SPY")]
        self.mapFreq = {'M':30, 'D':1}
        self.data = {}
        self.firstPriceOpen = firstPriceOpen

        PM.loadSeries(self.assets)
        PM.loadSeries(self.riskFree)

        foundMask = np.array([guideline in PM.cachedAssets for guideline in self.assets])
        if foundMask.all() == False:
            print("Couldnt find assets data, reallocating.")
            self.assets = np.array(self.assets)[foundMask]
            self.allocation = np.array(self.allocation)[foundMask]/np.array(self.allocation)[foundMask].sum()
            self.direction = np.array(self.direction)[foundMask]
            self.assets = [tuple(pair) for pair in self.assets.tolist()]

        if renormalize:
            self.allocation = np.array(self.allocation) / np.array(self.allocation).sum()

        print('[Allocated value]: ',sum(self.allocation))

    def setAllocation(self, allocation, direction):
        self.allocation = np.array(allocation)
        self.direction = np.array(direction)

    def setCAPMFactors(self, factors):
        self.factors = factors

    @staticmethod
    def toDate(obj):
        if type(obj) is str:
            return dateparser.parse(obj).date()
        if type(obj) is dt.datetime:
            return obj.date()
        if type(obj) is dt.date:
            return obj
        else:
            return obj

    def setDateRange(self, start=None, end=None):

        if start is not None:
            start = Portfolio.toDate(start)
        if end is not None:
            end = Portfolio.toDate(end)
        self.dateRange = (start, end)

    def setFrequency(self, freqDays):

        assert(type(freqDays) is str)
        self.freq = freqDays

    def makeRiskFree(self):

        rfDaily = PM.getRiskFree(self.data['assetsReturn'].index, self.riskFree).ffill()
        cumProd = (1+rfDaily).cumprod()
        onFreq = cumProd.reindex(self.pReturn.index).ffill()
        returnOnFreq = onFreq.pct_change().fillna(0.0)

        self.data["rfDaily"] = rfDaily
        self.data["rfOnFreq"] = returnOnFreq

    def getPrices(self):

        assets = list(self.assets) + list(self.factors)
        assets = list(pd.unique(assets))

        priceTable = PM.getAssetsAttribute(assets, 'adjustedclose', dateRange=self.dateRange).fillna(method='ffill')
        if self.firstPriceOpen:
            # Get the second row index
            day = list(priceTable.index)[1]
            # Get the open prices
            openPrice = PM.getAssetsAttribute(self.assets, 'open', dateRange=(day, day)).fillna(
                method='ffill')
            priceTable.iloc[0] = openPrice.iloc[0]
            print("First row prices swapped to second day open prices.")

        priceOnFreq = priceTable.resample("{0}".format(self.freq)).ffill() if self.freq != '1D' else priceTable
        self.data['price'] = priceTable
        self.data['priceOnFreq'] = priceOnFreq

    def makeFactors(self):

        priceTable = self.data['price'][self.factors]#PM.getAssetsAttribute(factors, 'adjustedclose', dateRange=self.dateRange)
        priceOnFreq = priceTable.reindex(self.pReturn.index).ffill()
        returnOnFreq = priceOnFreq.pct_change().fillna(0.0)
        self.data["factorsReturn"] = priceTable.pct_change().fillna(0.0)
        self.data["factorsReturnOnFreq"] = returnOnFreq

    def makeReturn(self):

        #priceTable = PM.getAssetsAttribute(self.assets, 'adjustedclose', dateRange=self.dateRange).fillna(method='ffill')
        # if self.firstPriceOpen:
        #     # Get the second row index
        #     day = list(priceTable.index)[1]
        #     # Get the open prices
        #     openPrice = PM.getAssetsAttribute(self.assets, 'open', dateRange=(day, day)).fillna(
        #         method='ffill')
        #     priceTable.iloc[0] = openPrice.iloc[0]
        #     print("First row prices swapped to second day open prices.")
        #
        # priceOnFreq = priceTable.resample("{0}".format(self.freq)).ffill() if self.freq != '1D' else priceTable
        priceTable = self.data['price'][self.assets]
        returnOnFreq = self.data['priceOnFreq'][self.assets].pct_change().replace([-1, np.inf, -np.inf], np.nan).fillna(0.0)

        self.data["assetsReturn"] = priceTable.pct_change().replace([-1, np.inf, -np.inf], np.nan).fillna(0.0)
        self.data["assetsReturnOnFreq"]  = returnOnFreq

    # TODO, sometimes data is corrupted and price is ZERO
    # TODO, fix it somehow, maybe ignore the return or interpolate if series of zeros is not too long
    def makePortfolioReturns(self):
        self.pReturn, cumcontributions = PM.getPortfolioReturns(self.data["assetsReturnOnFreq"] , self.allocation, self.direction)
        self.contributions = pd.DataFrame(cumcontributions, index=self.pReturn.index, columns = self.data["assetsReturnOnFreq"].columns).pct_change().fillna(0.0)

    def adjustIndex(self):
        correct = self.pReturn.index
        self.data["factorsReturnOnFreq"] = self.data["factorsReturnOnFreq"].reindex(correct).fillna(0.0)
        self.data["rfOnFreq"] = self.data["rfOnFreq"].reindex(correct).fillna(method='ffill')

    @timeit
    def make(self):

        self.getPrices()
        self.makeReturn()
        self.makePortfolioReturns()
        self.makeRiskFree()
        self.makeFactors()
        #self.adjustIndex()

    def yearlyUnderwater(self):
        return PM._drawdown(self.pReturn, int(np.round(self.annualizationFactor())), retUnderwater=True)

    def drawdown(self, windowFreq, top=5):
        return PM._drawdown(self.pReturn, windowFreq, top=top)

    def drawup(self, windowFreq, top=5):
        return PM._drawup(self.pReturn, windowFreq, top=top)

    def capm(self, intercept=True):

        portfolioReturnVec = self.pReturn[['return']].values
        rfVec = self.data["rfOnFreq"].loc[self.pReturn.index].values
        factors = self.data["factorsReturnOnFreq"].loc[self.pReturn.index].values

        excessReturn = portfolioReturnVec - rfVec
        excessFactors = factors - rfVec

        return PM._alphaBeta(excessReturn, excessFactors, intercept)

    def vol(self):
        return self.pReturn['return'].std() * np.sqrt(self.annualizationFactor())

    def expected(self):
        return self.pReturn['return'].mean() * self.annualizationFactor()

    def median(self):
        return self.pReturn['return'].median() * self.annualizationFactor()

    def plotUnderwater(self):

        fig, ax = plt.subplots(figsize=(13, 7))
        dd = np.round(self.yearlyUnderwater() * 100,2)
        dd.plot.area(ax=ax, alpha=0.5)
        ax.set_title("Yearly Underwater")
        ax.set_ylabel("Drawdown 1 year (%)")
        ax.set_xlabel("Date")
        fig.autofmt_xdate()

    def plotVAR(self,p):

        fig, ax = plt.subplots(figsize=(13, 7))
        returns = self.pReturn['return']
        percentile = np.percentile(returns, p)
        if p > 50:
            esf = returns[returns > percentile].mean()
        else:
            esf = returns[returns < percentile].mean()
        distplot(returns, ax=ax, norm_hist=False)
        ax.plot([percentile, percentile], [0,len(returns)/10],'r--',linewidth=2.0)
        ax.set_title("Distribution of Returns\n From {0} to {1}".format(self.pReturn.index[0].to_pydatetime().date(),self.pReturn.index[-1].to_pydatetime().date()))
        ax.set_xlabel("Returns {0}".format(self.freq))
        vals = ax.get_xticks()
        ax.set_xticklabels(['{:,.2%}'.format(x) for x in vals])

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = "P{0}: {1:.2%}\nESF{2}: {3:.2%}".format(p, percentile, p, esf)
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20,
                verticalalignment='top', bbox=props)

    def plotParametricVAR(self, p):

        fig, ax = plt.subplots(figsize=(13, 7))
        original = self.pReturn['return']
        mean = original.mean()
        std = original.std()
        returns = np.random.randn(5000) * std + mean

        percentile = np.percentile(returns, p)
        if p > 50:
            esf = returns[returns > percentile].mean()
        else:
            esf = returns[returns < percentile].mean()
        distplot(returns, ax=ax, norm_hist=False)
        ax.plot([percentile, percentile], [0, len(original)/10], 'r--', linewidth=2.0)
        ax.set_title("Normal parametric distribution of Returns\n From {0} to {1}".format(self.pReturn.index[0].to_pydatetime().date(),self.pReturn.index[-1].to_pydatetime().date()))
        ax.set_xlabel("Returns {0}".format(self.freq))
        vals = ax.get_xticks()
        ax.set_xticklabels(['{:,.2%}'.format(x) for x in vals])

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = "P{0}: {1:.2%}\nESF{2}: {3:.2%}".format(p, percentile, p, esf)
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20,
                verticalalignment='top', bbox=props)


    def toPct(self, series):
        return np.round(100*(series-1),2)

    def plot_like(self, returns, title=''):

        fig, ax = plt.subplots(figsize=(13, 7))
        formatter = FinancialDateFormatter(self.pReturn.index)
        xticks = np.arange(len(self.pReturn.index))
        ax.xaxis.set_major_formatter(formatter)

        values = np.cumprod(1 + returns)
        ax.plot(xticks, self.toPct(values), label='Portfolio')
        for factor in self.factors:
            xticks = np.arange(len(self.data["factorsReturnOnFreq"].index))
            ax.plot(xticks, self.toPct((1 + self.data["factorsReturnOnFreq"][factor]).cumprod()), label=factor[-1])
        ax.legend()
        ax.set_title(title)
        ax.set_ylabel("Total Return %")
        ax.set_xlabel("Date")
        ax.grid(True)
        plt.xticks(rotation=45)
        # fig.autofmt_xdate()
        fig.show()

    def plot(self,title='', multiplier=1.0):

        fig, ax = plt.subplots(figsize=(13,7))
        formatter = FinancialDateFormatter(self.pReturn.index)
        xticks = np.arange(len(self.pReturn.index))
        ax.xaxis.set_major_formatter(formatter)

        values = self.pReturn['return'].copy().values
        n = len(values)
        multiplier = np.power(multiplier, 1.0/n)
        values *= multiplier
        values = np.cumprod(1+values)
        ax.plot(xticks, self.toPct(values), label='Portfolio')
        for factor in self.factors:
            xticks = np.arange(len(self.data["factorsReturnOnFreq"].index))
            ax.plot(xticks, self.toPct((1+self.data["factorsReturnOnFreq"][factor]).cumprod()), label=factor[-1])
        ax.legend()
        ax.set_title(title)
        ax.set_ylabel("Total Return %")
        ax.set_xlabel("Date")
        ax.grid(True)
        plt.xticks(rotation=45)
        #fig.autofmt_xdate()
        fig.show()
        return values

    def annualizationFactor(self):

        mapFreq = {'M': 30, 'D': 1, 'W': 7}
        freqD = eval(self.freq.replace(self.freq[-1], "*{0}".format(mapFreq[self.freq[-1]])))
        convD_WD = (252/360)
        return 252 / (convD_WD * freqD) if self.freq != '1D' else 252

    # Put this inside Portfolio Methods
    @staticmethod
    def _sharpe(returns, freq):
        return (returns.mean()/returns.std())

    def sharpe(self):

        portfolioReturnVec = self.pReturn[['return']].values
        rfVec = self.data["rfOnFreq"].values
        excessReturn = portfolioReturnVec - rfVec

        return self._sharpe(excessReturn, self.freq) * np.sqrt(self.annualizationFactor())

    def cumreturn(self, wrt=None):
        if wrt is None:
            return self.pReturn['cumreturn'].tail(1).values[0] - 1.0
        else:
            data = self.pReturn.tail(wrt)
            return (data['cumreturn'].tail(1).values/data['cumreturn'].head(1).values)[0] - 1

    def cumreturnFactors(self,wrt=None):
        cumprod = (1+self.data["factorsReturnOnFreq"]).cumprod()
        if wrt is None:
            return (cumprod - 1).tail(1)
        else:
            data = cumprod.tail(wrt)
            return (data.tail(1).values/data.head(1).values) - 1

    def summarizeConstituents(self):

        returnData = self.data['assetsReturnOnFreq']
        factorsReturn = self.data['factorsReturnOnFreq']
        assets = list(returnData.columns)

        excessReturn = returnData.values - self.data['rfOnFreq'].values
        excessFactorsReturn = factorsReturn.values - self.data['rfOnFreq'].values
        annualizationFactor = self.annualizationFactor()

        cumReturns = (1+returnData).cumprod()
        finalReturn = cumReturns.tail(1).T
        elapsed =  (cumReturns.index[-1] - cumReturns.index[0]).days
        annualizedReturn = np.power(finalReturn, 365/elapsed)
        vol = returnData.std(0) * np.sqrt(annualizationFactor)
        expected = returnData.mean(0) * annualizationFactor
        sharpe = excessReturn.mean(0) * annualizationFactor / vol

        summaries = {}
        for asset in assets:
            summary = {}
            i = assets.index(asset)

            alpha, beta, errors = PM._alphaBeta(excessReturn[:,i], excessFactorsReturn)
            excessReturnMkt = returnData.values[:,i][None].T - factorsReturn.values[:,0][None].T

            eMktMean = excessReturnMkt.mean(0)
            eMktStd = excessReturnMkt.std(0)
            ir = (eMktMean/eMktStd) * np.sqrt(annualizationFactor)

            dd = PM._drawdown(pd.DataFrame(returnData.values[:,i], columns=['return']), int(np.round(self.annualizationFactor()))).values[0]

            summary['Total Return'] = finalReturn.loc[[asset]].values.ravel()[0] - 1
            summary['Annualized Return'] = annualizedReturn.loc[[asset]].values.ravel()[0] - 1
            summary['Expected Return'] = expected.loc[[asset]].values.ravel()[0]
            summary['Volatility'] = vol.loc[[asset]].values.ravel()[0]
            summary['Maximum Drawdown 1y'] = dd
            summary['Sharpe'] = sharpe.loc[[asset]].values.ravel()[0]
            summary['Alpha'] = alpha * annualizationFactor
            summary['Beta'] = beta
            summary['IR'] = ir
            summaries[asset[1]] = summary

        return summaries



    def informationRatio(self):

        pReturn = self.pReturn['return']
        market = self.data['factorsReturnOnFreq'].values[:,0]
        excess = pReturn - market
        ir = (excess.mean()/excess.std()) * self.annualizationFactor()
        return ir

    def summarize(self):

        annualizationFactor = self.annualizationFactor()
        alpha, beta, errors = self.capm()
        errorVol = errors.std() * np.sqrt(annualizationFactor)
        vol = self.vol()

        dds = self.drawdown(int(np.round(annualizationFactor)))
        max = dds.min()
        avg = dds.mean()
        sharpe = self.sharpe()
        ir = self.informationRatio()
        elapsed = (self.data['assetsReturnOnFreq'].index[-1] - self.data['assetsReturnOnFreq'].index[0]).days

        summary = {}
        summary['Alpha'] = alpha * annualizationFactor
        summary['Beta'] = beta
        summary['Expected'] = self.expected()
        summary['Vol'] = vol
        summary['Vol Idio'] = errorVol
        summary['Annualized Return'] = np.power(1+self.cumreturn(), 365/elapsed) - 1
        summary['Information Ratio'] = ir
        summary['Sharpe'] = sharpe
        summary['Maximum Drawdown 1y'] = max

        print("")
        print("[PORTFOLIO SUMMARY]")
        print("[FROM]: ", self.dateRange[0])
        print("[TO]: ", self.dateRange[1])
        print("")
        print("Sharpe ratio: ", sharpe)
        print("Alpha: ", alpha * annualizationFactor)
        print("Beta: ", beta)
        print("Expected: ", self.expected())
        print("Annualized return: ", summary['Annualized Return'])
        print("Vol: ", vol)
        print("Vol Idio: ", errorVol)
        print("Total Return: ", self.cumreturn())
        print("Information Ratio: ", ir)

        for factor in self.factors:
            volFactor = self.data["factorsReturnOnFreq"][factor].std() * np.sqrt(annualizationFactor)
            excessReturnFactor = (self.data["factorsReturnOnFreq"][factor].values - self.data["rfOnFreq"].values)
            sharpeFactor = self._sharpe(excessReturnFactor, self.freq) * np.sqrt(self.annualizationFactor())
            print("Vol {0}: ".format(factor), volFactor)
            print("Sharpe {0}: ".format(factor), sharpeFactor)

        print("Cumreturn factors: ",self.cumreturnFactors())
        print("Max 1y DD: ", max)
        print("Avg last 5 1y DD: ", avg)
        print("")
        return summary

class StrategyPortfolio(Portfolio):

    def __init__(self, backtestHistory):

        self.history = backtestHistory
        last = list(backtestHistory.keys())[-1]
        first = list(backtestHistory.keys())[0]
        firstPi = backtestHistory[first]['pi']
        lastPi = backtestHistory[last]['pi']

        assets = []
        for key in backtestHistory:
            assets += backtestHistory[key]['pi'].assets
        assets = pd.unique(assets)

        super().__init__(assets, np.ones(len(assets)), np.ones(len(assets)), firstPi.riskFree)

        # make routine
        self.factors = lastPi.factors
        self.setDateRange(firstPi.dateRange[0], lastPi.dateRange[1])
        self.getPrices()
        self.makeReturn()
        self.pReturn = self.aggRetuns()
        self.makeFactors()
        self.makeRiskFree()

    def aggRetuns(self):
        final = {}
        for key in list(self.history.keys())[::-1]:
            final.update(self.history[key]['pi'].pReturn.T.to_dict())
        sReturn = pd.DataFrame(final).T.sort_index()
        sReturn['cumreturn'] = (sReturn['return'] + 1).cumprod()
        return sReturn







