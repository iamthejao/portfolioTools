from portfolio import Portfolio
import numpy as np


# Just a wrapper to use the Portfolio class without caring about allocation and direction vectors
class Universe(Portfolio):

    def __init__(self, assets, riskFree):
        allocation = np.ones(len(assets))
        direction = np.ones(len(assets))
        super().__init__(assets, allocation, direction, riskFree)
