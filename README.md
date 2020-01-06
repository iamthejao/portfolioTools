# Portfolio Tools

Developed by Joao Augusto, under MIT Open License
Tool still in development and prone to bugs.

# Functionalities

The tool is able to download timeseries from the Alphavantage (https://www.alphavantage.co) API, build and save the timeseries in parquet files stored in disk that are then read and cached by a global portfolio module which provides support for the Portfolio classes. The module is able to calculate historical returns, many statistics and some risk parameters as VAR, Parametric VAR, drawdowns and others.

# How to download and build data
Set up JSON file like the ones in the json/ folder.
Insert alphavantage free key in the "alpha.key" file.

run:
python downloader.py json/nameoffile.json
output:

Using key:  KEYNUMBER
Building ticker: TIME_SERIES_DAILY_ADJUSTED - IWV
Wrong input format on mkDir, did not make. (Ignore this warning)
Building ticker: TIME_SERIES_DAILY_ADJUSTED - SPY
Wrong input format on mkDir, did not make.
Building ticker: TIME_SERIES_DAILY_ADJUSTED - QQQ
Wrong input format on mkDir, did not make.
Building ticker: TIME_SERIES_DAILY_ADJUSTED - OILU
Wrong input format on mkDir, did not make.
Ready: 100.00%

run:
python series_handler.py json/nameoffile.json
output:
progress bar will show.

The first command will download the timeseries and save it as json files
The second command will read the saved files, calculate new columns as return & others and save as parquet files (used by the module)

# How to run a portfolio

In order to facilitate the tutorials, the timeseries are already downloaded in this repository.

Please check notebook tutorials.

