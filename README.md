# Stock-Predition

Project Overview
This project implements and compares three deep learning models (RNN, LSTM, and GRU) for predicting Apple Inc. (AAPL) stock prices using historical time-series data from Yahoo Finance. The optimized GRU model achieved the best performance with RMSE of 7.3 and MAE of 6.0 on test data.

Requirements
Python 3.7+

TensorFlow 2.x

Keras

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

Scipy

Statsmodels

Dataset
Source: Yahoo Finance (AAPL historical data)

Time period: September 1, 2015 to November 9, 2023 (2,063 samples)

Features: Date, Open, High, Low, Close, Adjusted Close, Volume

Data file: AAPL.csv (included in repository)

Implementation Steps
1. Data Preprocessing
Handling missing values

Outlier detection using Z-scores

Feature normalization with StandardScaler

Dataset splitting (60% train, 20% validation, 20% test)

Time-series windowing (40-day lookback period)

2. Model Architectures
RNN Model
3 stacked SimpleRNN layers (50 units each)

Tanh activation

Adam optimizer (learning rate=0.0005)

MSE loss function

LSTM Model
2 LSTM layers (16 units each)

Dropout (0.2)

Linear activation in dense layer

Adam optimizer with MSE loss

GRU Model (Base)
3 GRU layers (32 units)

Adam optimizer with MSE loss

Optimized GRU Model
Added dropout (0.2)

Learning rate scheduler (ReduceLROnPlateau)

Increased units to 64 per layer

Early stopping

3. Training
Batch size: 64

Epochs: 200

Validation split: 0.25

Callbacks: Early stopping and learning rate reduction

4. Evaluation Metrics
RMSE (Root Mean Squared Error)

MSE (Mean Squared Error)

MAE (Mean Absolute Error)

How to Run
Install required packages: pip install -r requirements.txt

Place dataset in data/ folder

Run Jupyter notebook: jupyter notebook stock_prediction.ipynb

Execute cells sequentially

Results
Model	MAE	MSE	RMSE
RNN	10.8	172.6	13.1
LSTM	15.2	366.7	19.1
GRU	7.3	73.7	8.5
Optimized GRU	6.0	53.5	7.3


Key Findings
GRU outperformed both RNN and LSTM models

Model optimization techniques (dropout, LR scheduling, increased units) improved performance

The simplified gate structure of GRU proved effective for this stock prediction task
