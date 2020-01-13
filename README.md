# Last Update
Fixed a bug inside the Parametric and NonParametric VAR function

# Portfolio Tools

Developed by Joao Augusto, under MIT Open License
Tool still in development and prone to bugs.

# Functionalities

The tool is able to download timeseries from the Alphavantage (https://www.alphavantage.co) API, build and save the timeseries in parquet files stored in disk that are then read and cached by a global portfolio module which provides support for the Portfolio classes. The module is able to calculate historical returns, many statistics and some risk parameters as VAR, Parametric VAR, drawdowns and others.

# How to download and build data
Set up JSON file like the ones in the json/ folder.
Insert alphavantage free key in the "alpha.key" file.

run:
python downloader.py json/nameoffile.json and then python series_handler.py json/nameoffile.json

The first command will download the timeseries and save it as json files
The second command will read the saved files, calculate new columns as return & others and save as parquet files (used by the module)

# How to run a portfolio

In order to facilitate the tutorials, the timeseries are already downloaded in this repository.

Please check notebook tutorials for portfolio statistics, risk measures and backtest. I recommend first checking the Portfolio Tutorial Notebook.

# How to use data from other sources (like BBG backtest)
A notebook for reading excel and transforming into parquet file is provided along with the tutorials. The timeseries can then be loaded normally with the Portfolio class.

# Portfolio Optimization
The file portfolio_optimizer provides a parallel implementation of a population based meta-heuristic algorithm known as Particle Swarm Optimization. This algorithm provides robust results for the optimisation of multidimensional constrained functions. One of the functions implemented treats the portfolio selection + allocation problem as a single optimisation problem and is able to provide algorithmically picked portfolios that follow certain optimisation criteria and constraints from a universe of stocks in a defined time period. The PortfolioOptimizer can be used along with the Backtest class for backtesting and optimising strategies, mainly momentum strategy given the information we have available but can be extended for other types of strategies.

No tutorial notebook is provided yet given this would take more effort to explain and implement.
