# Time Series Forecasting in Python

A comprehensive tutorial on time series analysis, transformation, and forecasting using real-world air quality data from Beijing. This project walks through the complete pipeline -- from exploratory analysis and stationarity testing to building and comparing 20+ forecasting models.

---

## Features

- End-to-end time series workflow using real-world pollution data (Beijing PM2.5 dataset, 2014--2019)
- Time series decomposition: level, trend, seasonality, and noise
- Stationarity analysis with ACF/PACF plots, rolling statistics, and the Dickey-Fuller test
- Data transforms: differencing, log scaling, smoothing, and moving averages
- 20+ forecasting models implemented and benchmarked side by side
- Ensemble methods combining XGBoost, LightGBM, and TensorFlow for best results
- Evaluation using MAE, MAPE, RMSE, and R-squared metrics
- Model interpretability with SHAP values
- Automated dataset download and preprocessing script

## Tech Stack

- **Language:** Python 3.7
- **Notebooks:** Jupyter
- **Core Libraries:** pandas, NumPy, SciPy, matplotlib, statsmodels
- **Classical Models:** statsmodels (AR, MA, ARMA, ARIMA, SARIMA, SES, HWES), pmdarima (Auto-ARIMA)
- **Machine Learning:** scikit-learn (Bayesian Ridge, Lasso, SVM, Random Forest, KNN), XGBoost, LightGBM
- **Deep Learning:** TensorFlow/Keras (LSTM), GluonTS with MXNet (DeepAR)
- **Other:** Facebook Prophet, Bayesian Optimization, SHAP
- **Environment:** Conda (environment.yml) or pip (requirements.txt)

## Project Structure

```
time-series-forecasting-python/
|-- 01-Analysis&transforms.ipynb       # Data exploration, decomposition, stationarity tests
|-- 02-Forecasting_models.ipynb        # Model training and prediction
|-- 03-Results_analysis&discussion.ipynb  # Results comparison and discussion
|-- time-series-forecasting-tutorial.ipynb  # Combined tutorial notebook
|-- datasets/
|   |-- air_pollution.csv              # Preprocessed Beijing air quality data
|   |-- training.csv                   # Training split
|   |-- test.csv                       # Test split
|   |-- international_airline_passengers.csv  # Additional example dataset
|   |-- download_datasets.py           # Script to download and preprocess data
|-- utils/
|   |-- metrics.py                     # MAE, MAPE, RMSE, R2 evaluation utilities
|   |-- plots.py                       # Visualization helper functions
|-- results/                           # Output plots and figures
|-- docs/                              # Setup guide and documentation
|-- environment.yml                    # Conda environment specification
|-- requirements.txt                   # pip dependencies
```

## Getting Started

### Prerequisites

- Python 3.7
- Conda (recommended) or pip
- Jupyter Notebook or JupyterLab

### Installation

**Option 1: Conda (recommended)**

```bash
git clone https://github.com/akshay1389/time-series-forecasting-python.git
cd time-series-forecasting-python
conda env create -f environment.yml
conda activate time
```

**Option 2: pip**

```bash
git clone https://github.com/akshay1389/time-series-forecasting-python.git
cd time-series-forecasting-python
pip install -r requirements.txt
```

### Download the Dataset

```bash
python datasets/download_datasets.py
```

This script automatically downloads the Beijing air quality dataset from the UCI Machine Learning Repository and preprocesses it into daily frequency.

### Run the Notebooks

```bash
jupyter notebook
```

Open the notebooks in order:
1. `01-Analysis&transforms.ipynb` -- Explore the data and apply transforms
2. `02-Forecasting_models.ipynb` -- Train and evaluate forecasting models
3. `03-Results_analysis&discussion.ipynb` -- Compare results and analyze performance

Alternatively, use the combined `time-series-forecasting-tutorial.ipynb` for the full walkthrough.

## Usage

The notebooks are structured as a tutorial. Each one builds on the previous:

**Notebook 1 -- Analysis and Transforms:** Load the Beijing air quality dataset, decompose the time series into trend/seasonality/residual components, test for stationarity using Dickey-Fuller, and apply transforms (differencing, log, smoothing) to achieve stationarity.

**Notebook 2 -- Forecasting Models:** Train models ranging from classical statistical methods (AR, ARIMA, SARIMA) through machine learning (Random Forest, XGBoost, LightGBM, SVM) to deep learning (LSTM, DeepAR, Prophet). Both univariate and multivariate (using weather features like temperature and pressure) approaches are tested.

**Notebook 3 -- Results and Discussion:** Compare all models using MAE, MAPE, RMSE, and R2. The best-performing approach is an ensemble of LightGBM and TensorFlow LSTM (MAE: 27.34, R2: 0.77), demonstrating that combining gradient boosting with deep learning yields strong results on this dataset.

The utility modules in `utils/` provide reusable functions for evaluation metrics and plotting that can be imported into your own projects.

## License

This project is open source. See the repository for license details.
