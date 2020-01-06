import pandas as pd
from multiprocessing import Pool, RawArray, cpu_count
from portfolio_methods import PortfolioMethods
import numpy as np
import copy
from decorators import timeit
from sklearn.linear_model import LinearRegression
import numpy.random as npr
import json
import datetime as dt
from portfolio import PM, Portfolio


GLOBAL_DICT = {}
def init_worker(returnMatrix, excessReturnMatrix, excessMarketReturn, dates, shapes):

    global GLOBAL_DICT
    GLOBAL_DICT['returnMatrix'] = np.frombuffer(returnMatrix).reshape(shapes['return'])
    GLOBAL_DICT['excessReturnMatrix'] = np.frombuffer(excessReturnMatrix).reshape(shapes['return'])
    GLOBAL_DICT['excessMarketReturn'] = np.frombuffer(excessMarketReturn).reshape(shapes['market'])
    GLOBAL_DICT['dates'] = dates

def test_global(i):
    global GLOBAL_DICT
    return GLOBAL_DICT['returnMatrix']


class PortfolioOptimizer:

    @timeit
    def __init__(self, universeObj):

        self.universeObj = universeObj
        self.universeObj.make()
        self.init()

    @timeit
    def init(self):

        self.returnCols = self.universeObj.assets
        self.returns = self.universeObj.data['assetsReturnOnFreq']
        self.riskFree = self.universeObj.data['rfOnFreq']
        self.market = self.universeObj.data['factorsReturnOnFreq']
        self.makeMatrices()

    @timeit
    def makeMatrices(self):

        self.returnMatrix = self.returns.values
        self.rfVector = self.riskFree.values
        self.excessReturn = self.returnMatrix - self.rfVector
        self.marketVec = self.market.values
        self.excessMarketReturn = self.marketVec - self.rfVector

        self.shapes = {'return': self.returnMatrix.shape,
                       'market': self.excessMarketReturn.shape}

    @timeit
    def getRawArrays(self):
        returnMatrix = RawArray('d', self.returnMatrix.reshape(np.prod(self.shapes['return'])))
        excessReturnMatrix = RawArray('d', self.excessReturn.reshape(np.prod(self.shapes['return'])))
        excessMarketReturn = RawArray('d', self.excessMarketReturn.reshape(np.prod(self.shapes['market'])))
        return returnMatrix, excessReturnMatrix, excessMarketReturn

    @staticmethod
    def calculateAllocation(vector):

        #pickSize = max([int(abs(vector[-1])), 1])
        pickSize = 30
        vector = np.abs(vector)#np.abs(np.array(vector[:-1]))##

        # Selecting stocks, idx of the # pickSize biggest
        idx = (-vector).argsort()[:pickSize]
        weights = vector[idx]
        allocation = weights/weights.sum()
        direction = np.ones(pickSize)

        return idx, allocation, direction

    @staticmethod
    def calculateAllocationLongShort(vector):

        #pickSize = max(np.int(np.abs(vector[-1])), 10)
        #pickSize = min(pickSize, 30)
        pickSize = 30

        #vector = vector[:-1]#np.array(vector,dtype=np.float32)[:-1]
        direction = (vector > 0).astype(np.float32)
        direction[direction == 0] = -1

        allocWeight = np.abs(vector)

        # Selecting stocks, idx of the # pickSize biggest
        idx = ((-allocWeight).argsort()[:pickSize])
        weights = allocWeight[idx]
        allocation = (weights / weights.sum())
        direction = direction[idx]

        return idx, allocation, direction

    @staticmethod
    def fitness_print(vector):
        return PortfolioOptimizer.fitness_test1(vector, printData=True)

    @staticmethod
    def fitness_strategy1(vector, printData=False):

        global GLOBAL_DICT

        idx, allocation, direction = PortfolioOptimizer.calculateAllocationLongShort(vector)
        if (allocation > 0.3).any() and printData == False:
            return 0

        selectedReturn = GLOBAL_DICT['returnMatrix'][:, idx]
        selectedExcessReturn = GLOBAL_DICT['excessReturnMatrix'][:, idx]
        marketReturn = GLOBAL_DICT['excessMarketReturn']
        dates = GLOBAL_DICT['dates']

        pReturn = PM._pReturn(selectedReturn.astype(np.float32), allocation, direction)
        pReturnExcess = PM._pReturn(selectedExcessReturn.astype(np.float32), allocation, direction)

        mktError = (pReturnExcess - marketReturn)
        trackingError = np.std(mktError)

        # Portfolio Metrics
        alpha, beta = PM._alphaBeta(pReturnExcess, marketReturn, intercept=True)
        beta = np.sum(beta)
        targetBeta = 0.2
        betaPenalty = np.abs(targetBeta - beta)

        volMkt = marketReturn.std()
        vol = pReturn.std()

        mean = pReturnExcess.mean()

        maximizeTerm = (1.0 + alpha + pReturnExcess.mean()) * np.sign(alpha)
        minimizeTerm = (1 + volMkt + betaPenalty/10 + trackingError/10)

        fitness = maximizeTerm / minimizeTerm
        return fitness

    @staticmethod
    def fitness_strategy2(vector, printData=False):

        global GLOBAL_DICT

        idx, allocation, direction = PortfolioOptimizer.calculateAllocation(vector)
        if (allocation > 0.5).any() and printData == False:
            return 0

        selectedReturn = GLOBAL_DICT['returnMatrix'][:, idx]
        selectedExcessReturn = GLOBAL_DICT['excessReturnMatrix'][:, idx]
        marketReturn = GLOBAL_DICT['excessMarketReturn']
        dates = GLOBAL_DICT['dates']

        pReturn = PM._pReturn(selectedReturn.astype(np.float32), allocation, direction)
        pReturnExcess = PM._pReturn(selectedExcessReturn.astype(np.float32), allocation, direction)

        mktError = (pReturnExcess - marketReturn)
        trackingError = np.std(mktError)

        # Portfolio Metrics
        alpha, beta = PM._alphaBeta(pReturnExcess, marketReturn, intercept=True)
        beta = np.sum(beta)
        #targetBeta = 0.5
        betaPenalty = 0#np.abs(targetBeta - beta)

        volMkt = marketReturn.std()
        vol = pReturn.std()

        mean = pReturnExcess.mean()

        maximizeTerm = (1.0 + alpha + pReturnExcess.mean())
        minimizeTerm = (1 + volMkt + betaPenalty/10 + trackingError/10)

        fitness = maximizeTerm / minimizeTerm
        return fitness

    @staticmethod
    def fitness_best_sharp(vector):

        global GLOBAL_DICT
        idx, allocation, direction = PortfolioOptimizer.calculateAllocationLongShort(vector)

        if (allocation > 0.4).any():
            return 0

        selectedReturn = GLOBAL_DICT['returnMatrix'][:, idx]
        selectedExcessReturn = GLOBAL_DICT['excessReturnMatrix'][:, idx]
        marketReturn = GLOBAL_DICT['excessMarketReturn']
        dates = GLOBAL_DICT['dates']

        cumulative, _ = PM._pReturn(selectedExcessReturn.astype(np.float32), allocation, direction)
        pReturnExcess = np.concatenate(([0], np.diff(cumulative)/cumulative[:-1]))

        mean = pReturnExcess.mean()
        std = pReturnExcess.std()
        sharpe = mean/std
        alpha, beta, error = PM._alphaBeta(pReturnExcess, marketReturn, intercept=True)
        vol = pReturnExcess.std()
        return sharpe

    @staticmethod
    def fitness_test1(vector, printData=False):

        global GLOBAL_DICT
        idx, allocation, direction = PortfolioOptimizer.calculateAllocationLongShort(vector)

        if (allocation > 0.3).any() and printData == False:
            return 0

        # Now the dataframe will be split and fitness summed over
        aggFitness = 0

        selectedReturn = GLOBAL_DICT['returnMatrix'][:, idx]
        selectedExcessReturn = GLOBAL_DICT['excessReturnMatrix'][:, idx]
        marketReturn = GLOBAL_DICT['excessMarketReturn']
        dates = GLOBAL_DICT['dates']

        start = 0
        sliceSize = 999999
        count = 0
        targetBeta = 0.0

        while (start < len(selectedReturn)):

            sliceReturn = selectedReturn[start:start+sliceSize]
            sliceExcessReturn = selectedExcessReturn[start:start + sliceSize]
            sliceMarket = marketReturn[start:start + sliceSize]
            sliceDates = dates[start:start + sliceSize]

            pReturn = PM._pReturn(sliceReturn.astype(np.float32), allocation, direction)
            pReturnExcess = PM._pReturn(sliceExcessReturn.astype(np.float32), allocation, direction)

            mktError = (pReturnExcess - sliceMarket)
            trackingError = np.std(mktError)

            # Portfolio Metrics
            alpha, beta = PM._alphaBeta(pReturnExcess, sliceMarket, intercept=True)
            beta = np.sum(beta)

            volMkt = sliceMarket.std()
            vol = pReturn.std()

            #if vol < 0.3 * volMkt:
            #    return 0.0

            mean = pReturnExcess.mean()

            dd = PM._drawdown(pd.DataFrame(pReturn, columns=['return']), window_size=52, top=2)
            du = PM._drawup(pd.DataFrame(pReturn, columns=['return']), window_size=4, top=10)
            avgDU = np.mean(np.abs(du.values))

            avgDD = np.mean(np.abs(dd.values))
            maxDD = np.abs(dd.values[0])

            maximizeTerm = (1.0 + alpha + pReturnExcess.mean()) * np.sign(alpha)
            minimizeTerm = (1 + volMkt + np.abs(beta - targetBeta))

            fitness = maximizeTerm / minimizeTerm
            aggFitness += fitness
            count += 1
            start += sliceSize

            if printData:
                print("Alpha: ",alpha)
                print("Beta: ", beta)
                print("Vol: ", vol)
                print("Vol market: ", volMkt)
                print("AvgDD: ", avgDD)
                print("maxDD: ", maxDD)

        return aggFitness/count

    @timeit
    def initPool(self):
        self.pool = Pool(cpu_count()+2, initializer=init_worker,
                         initargs=list(self.getRawArrays()) + [self.returns.index, self.shapes])

    @timeit
    def pso_optimize(self,fitness,
                     parameters=(0.74, 1.63, 1.63),
                     iterations=100,
                     particles=70,
                     init_range=(-30, 30),
                     init_velocity=0.1,
                     seed=None,
                     save_every=10, debug=False,
                     insert_solution=None,
                     chunksize=1,
                     dimension=None):

        self.initPool() if debug is False else None

        gbestsSave = []

        npr.seed(seed)
        dimension = len(self.returnCols) if dimension is None else dimension

        # Parameters initialization
        inertia_weight = parameters[0]
        alfa = (inertia_weight - 0.4) / iterations
        c1 = parameters[1]
        c2 = parameters[2]

        # Initializing population & speed
        population = ((init_range[1] - init_range[0]) * npr.random((particles, dimension))) + init_range[0]
        pbest = population.copy()

        velocity = init_velocity * (
                ((init_range[1] - init_range[0]) * npr.random((particles, dimension))) + init_range[0])

        pFitness = np.array(self.pool.map(fitness, population, chunksize=chunksize) if debug is False else list(map(fitness, population)))

        # MAXIMIZATION
        gbestFitness = pFitness.max()
        gbest = population[pFitness.argmax()]
        pbestFitness = pFitness.copy()

        # Iterations

        for i in range(0, iterations):

            # Update velocity
            velocity = (inertia_weight * velocity) \
                       + c2 * npr.random((particles, dimension)) * (
                    (gbest * np.ones((particles, 1))) - population) \
                       + c1 * npr.random((particles, dimension)) * (
                               pbest - population)

            # Update population
            population += velocity

            # No Boundary checks here.
            pFitness = np.array(self.pool.map(fitness, population, chunksize=chunksize) if debug is False else list(map(fitness, population)))

            # Update gbest
            pArgMax = pFitness.argmax()
            if gbestFitness < pFitness[pArgMax]:
                gbestFitness = pFitness[pArgMax]
                gbest = population[pArgMax]

            # Update Pbest
            mask = (pFitness > pbestFitness).ravel()
            pbestFitness[mask] = pFitness[mask]
            pbest[mask, :] = population[mask, :]

            inertia_weight -= alfa
            if (i % save_every == 0):
                print("Gbest fitness {0}, current iteration {1}".format(gbestFitness, i))
                gbestsSave.append(gbest)

            # Checking if solution should be inserted into population
            if insert_solution is not None and type(insert_solution) is dict:
                insert = False
                if 'fitness' in insert_solution.keys():
                    if insert_solution['fitness'] < pbestFitness.mean():
                        insert = True
                if 'iteration' in insert_solution.keys():
                    if insert_solution['iteration'] <= i:
                        insert=True
                if insert:
                    fit = fitness(insert_solution['vector'])
                    idx = pbestFitness.argmin()
                    population[idx, :] = insert_solution['vector']
                    pbest[idx, :] = insert_solution['vector']
                    pbestFitness[idx] = fit
                    pFitness[idx] = fit
                    if fit > gbestFitness:
                        gbestFitness = fit
                        gbest = insert_solution['vector']
                    insert_solution = None

        #self.pool.map(PortfolioOptimizer.fitness_print, [gbest])
        return gbest, gbestFitness, pbest

if __name__ == '__main__':

    tickers = json.load(open("sp500.json", "r"))
    tickers = {'TIME_SERIES_DAILY_ADJUSTED': tickers['TIME_SERIES_DAILY_ADJUSTED']}
    tickers_flat = PortfolioOptimizer._flatDict(tickers)

    start = dt.datetime(2015, 1, 1)
    obj = PortfolioOptimizer(tickers_flat, ('RISK_FREE', 'USA'), start)

    # Starting global dict for debugging purpose
    init_worker(*(list(obj.getRawArrays())+[obj.returns.index, obj.shapes]))
    print(GLOBAL_DICT)

    obj.pso_optimize(PortfolioOptimizer.fitness_test1, iterations=10, debug=True)



