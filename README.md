# Adaptive Multi-task Learning for Probabilistic Load Forecasting

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](/AMRC_Python) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](#support-and-author)

This repository is the official implementation of the code developed in the paper "Adaptive Multi-task Learning for Probabilistic Load Forecasting".

The proposed method is a multi-task learning method for online and probabilistic load forecasting. The method can dynamically adapt to changes in consumption patterns and correlations among multiple entities. The techniques presented provide accurate probabilistic predictions for loads of multiples entities and asses load uncertainties.


## Implementation of the method

* main.py is the main file. In such file we can modify the values of hyper-parameters such as 
* model.py is the file that contains the functions to train and evaluate the model:
	- `initialize` initializes the parameters of the model.
	- `update_parameters` recursively updates the parameters of the model.
	- `update_model` updates the model each time new samples arrive.
	- `prediction` obtains the multi-task probabilistic forecasts in terms of multivariate Gaussian distributions.
	- `test` evaluates the forecasts in terms of RMSE and MAPE.
	- `adapt_covariance` simplifies the covariance matrix setting low values to zero to avoid spurious correlations.


## Installation 

```console
git clone https://github.com/onintzezaballa/Multitask-Load-Forecasting

```

To train and evaluate the model in the paper, run this command:

```console
python main.py

```

## Data

We use 4 publicly available datasets containing multiple entities
* [GEFCom2017](https://www.sciencedirect.com/science/article/abs/pii/S016920701930024X)
* [NewEngland](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info)
* [PJM](https://dataminer2.pjm.com/feed/hrl_load_estimated/definition)
* [Australia](https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data)

We save the data in .mat files with hourly load time series, temperature time series, and calendar-related information, such as the hour of day or day of the week.

## Test case

We display in this repository an example of the dataset [GEFCom2017](https://www.sciencedirect.com/science/article/abs/pii/S016920701930024X) with 8 entities.

## Support and Author

Onintze Zaballa

ozaballa@bcamath.org

[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://github.com/onintzezaballa)

## License 

MIT license.

