from portfolio import Portfolio, PM
import datetime as dt
from collections import OrderedDict
import utility
import copy
import numpy as np

class Backtester:

    def __init__(self, universeObj, start=None, end=None):

        if start is None:
            start = universeObj.dateRange[0]
        if end is None:
            end = universeObj.dateRange[1]

        self.universe = universeObj
        # Can still be None, None, which will be checked later
        self.dateRange = [start, end]
        if self.dateRange[0] is None or self.dateRange[1] is None:
            start, end = PM.getPortfolioDateRange(universeObj.assets)
            if self.dateRange[0] is None:
                self.dateRange[0] = start
            if self.dateRange[1] is None:
                self.dateRange[1] = end
        self.dateRange = tuple(copy.deepcopy(self.dateRange))

    def load(self):

        self.universe.make()
        self.originals = copy.deepcopy(self.universe.data)

    def reduce(self, frame):
        return frame.loc[(frame.index >= self.visibility[0]) & (frame.index <= self.visibility[1])]

    def reduceVisibility(self, dateRange):

        self.visibility = (np.datetime64(dateRange[0]), np.datetime64(dateRange[1]))
        for key in self.universe.data.keys():
            self.universe.data[key] = self.reduce(self.originals[key].copy())


    def backtest(self, strategy, frequency, daysBeforeVisibility=360, inverse=False):

        """
        Strategy function is a function that receives universeObj, start and end and should return
        assets, allocation, direction vectors

        frequency is the jump

        daysBeforeVisibility is the start and currentDay the end of which the strategy function should see
        """

        history = OrderedDict()
        firstDayData = self.dateRange[0] - dt.timedelta(days=daysBeforeVisibility)
        self.universe.dateRange = (firstDayData, self.dateRange[1])
        print("Loading universe data from {0} to {1}.".format(self.universe.dateRange[0], self.universe.dateRange[1]))
        self.load()
        print("Backtesting from {0} to {1}".format(self.dateRange[0], self.dateRange[1]))

        prevPi = None
        kwdict = {}
        currentDay = self.dateRange[0]
        while currentDay < self.dateRange[1]:

            startVisibility = currentDay - dt.timedelta(days=daysBeforeVisibility)
            self.reduceVisibility((startVisibility, currentDay))

            print("")
            print('[BACKTESTER]: Running strategy with data from: \n {0} to {1}'.format(startVisibility, currentDay))

            assets, allocation, direction, kwdict = strategy(self.universe, kwdict)

            if inverse:
                direction *= -1.0
            nextRun = min(utility.next_weekday(currentDay + dt.timedelta(days=frequency)), self.dateRange[1])

            print('[BACKTESTER]: Checking Portfolio from: \n {0} to {1}'.format(currentDay, nextRun))
            print("")

            pi = Portfolio(assets, allocation, direction, self.universe.riskFree)
            if prevPi is None:
                pi.setDateRange(currentDay, nextRun)
            else:
                pi.setDateRange(prevPi.pReturn.index.max(), nextRun)

            pi.factors = self.universe.factors
            pi.make()
            pi.summarize()


            history[currentDay] = {'assets': assets,
                                   'n_assets': len(assets),
                                   'allocation': allocation,
                                   'direction': direction,
                                   'pi': pi}

            currentDay = nextRun
            prevPi = pi

        return history
