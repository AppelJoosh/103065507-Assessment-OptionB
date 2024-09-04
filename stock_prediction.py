# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# YouTube link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf

# importing datetime to validate database start/end args
import datetime

# P1 specific imports for load_data()
from sklearn.model_selection import train_test_split
from collections import deque
from yahoo_fin import stock_info as si

from sklearn import preprocessing
# removed ".models" from statement below
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

# task3 imports
import talib
import mplfinance as fplt
import plotly

# ------------------------------------------------------------------------------
# Load Data
# TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
# ------------------------------------------------------------------------------
# DATA_SOURCE = "yahoo"
COMPANY = 'CBA.AX'

TRAIN_START = '2020-01-01'  # Start date to read
TRAIN_END = '2023-08-01'  # End date to read

# data = web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END) # Read data using yahoo

# moved to top
# import yfinance as yf

# Get the data for the stock AAPL
data = yf.download(COMPANY, TRAIN_START, TRAIN_END)

# ------------------------------------------------------------------------------
# Prepare Data
# To do:
# 1) Check if data has been prepared before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Use a different price value e.g. mid-point of Open & Close
# 3) Change the Prediction days
# ------------------------------------------------------------------------------
PRICE_VALUE = "Close"

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# Note that, by default, feature_range=(0, 1). Thus, if you want a different 
# feature_range (min,max) then you'll need to specify it here
scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1))
# Flatten and normalise the data
# First, we reshape a 1D array(n) to 2D array(n,1)
# We have to do that because sklearn.preprocessing.fit_transform()
# requires a 2D array
# Here n == len(scaled_data)
# Then, we scale the whole array to the range (0,1)
# The parameter -1 allows (np.)reshape to figure out the array size n automatically 
# values.reshape(-1, 1) 
# https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
# When reshaping an array, the new shape must contain the same number of elements 
# as the old shape, meaning the products of the two shapes' dimensions must be equal. 
# When using a -1, the dimension corresponding to the -1 will be the product of 
# the dimensions of the original array divided by the product of the dimensions 
# given to reshape to maintain the same number of elements.

# Number of days to look back to base the prediction
PREDICTION_DAYS = 60  # Original

# To store the training data
x_train = []
y_train = []

scaled_data = scaled_data[:, 0]  # Turn the 2D array back to a 1D array
# Prepare the data
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x - PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
# is an array of p inputs with each input being a 2D array 

# ------------------------------------------------------------------------------
# Build the Model
# TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
# ------------------------------------------------------------------------------
model = Sequential()  # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# This is our first hidden layer which also specifies an input layer.
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For some advances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.

# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more, the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If your training
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately make changes to your model
# (a source of overfitting). Thus, we do this in batches: We'll look at
# the aggregated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

# ------------------------------------------------------------------------------
# Test the model accuracy on existing data
# ------------------------------------------------------------------------------
# Load the test data
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

# test_data = web.DataReader(COMPANY, DATA_SOURCE, TEST_START, TEST_END)

test_data = yf.download(COMPANY, TEST_START, TEST_END)

# The above bug is the reason for the following line of code
# test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the first
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

model_inputs = scaler.transform(model_inputs)


# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way


# task B.2
def load_data(ticker, ds_start="2023-08-02", ds_end="2024-07-02", steps=50, scale=True, scale_type="minmax",
              shuffle=True, lookup_steps=1, split_by_date=True, test_size=0.2,
              feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    """
        comment block taken from P1 for readability, added own parameters
        Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
        Params:
            ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
            ds_start (str, YYYY-MM-DD format): set start date for ticker to download from. Default is 2023-08-02
            ds_end (str, YYYY-MM-DD format): set end date for ticker to download from. Default is 2024-07-02
            steps (int): the historical sequence length (i.e window size) used to predict, default is 50
            scale (bool): whether to scale prices from 0 to 1, default is True
            scale_type (str/preprocessing.[scale type]Scaler): the scaler you want to load. Ex. MinMax, MaxAbs, Robust.
                Default is minmax
            shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
            lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
            split_by_date (bool): whether we split the dataset into training/testing by date, setting it
                to False will split datasets in a random way
            test_size (float): ratio for test data, default is 0.2 (20% testing data)
            feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
        """

    # validate ds_start/ds_end
    try:
        datetime.date.fromisoformat(ds_start)
    except ValueError:
        raise ValueError("ds_start should be YYYY-MM-DD")
    try:
        datetime.date.fromisoformat(ds_end)
    except ValueError:
        raise ValueError("ds_end should be YYYY-MM-DD")

    # check if ticker has been loaded already
    if isinstance(ticker, str):
        # load from yfinance?
        # no, needs yahoo_fin's stock_info to get the columns for the first assert below
        df = si.get_data(ticker, ds_start, ds_end)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use directly
        df = ticker
    else:
        raise TypeError("ticker must be string or pd.DataFrame")

    # contains all elements to return
    result = {}
    # also return dataframe itself
    result['df'] = df.copy()

    # assert feature_columns are in dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in dataframe"

    # add date as a column
    if "date" not in df.columns:
        df["date"] = df.index

    # scales the data
    # minmax scaling
    if scale:
        column_scaler = {}
        # this SHOULD select from different scales, or load an existing scaler)
        if isinstance(scale_type, str):
            if scale_type.lower() == "minmax":
                scaler = preprocessing.MinMaxScaler()
            elif scale_type.lower() == "maxabs":
                scaler = preprocessing.MaxAbsScaler()
            elif scale_type.lower() == "robust":
                scaler = preprocessing.RobustScaler()
            else:
                scaler = preprocessing.StandardScaler()
        elif isinstance(scale_type, (preprocessing.MinMaxScaler, preprocessing.MaxAbsScaler,
                                     preprocessing.RobustScaler, preprocessing.StandardScaler)):
            scaler = scale_type
        else:
            raise TypeError("scale_type must be string or preprocessing scaler")

        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the scaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_steps)

    # last `lookup_step` columns contains NaN in future column
    # get them before dropping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_steps))

    # drop NaNs
    # removes NaNs from the dataframe if they exist
    df.dropna(inplace=True)

    # sequence_data and its related code lines are needed to construct X and y for split_by_date()
    sequence_data = []
    sequences = deque(maxlen=steps)

    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence

    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # checks if split_by_date in args is true
    # need to find a way to pass in specific dates, code doesn't seem to use them?
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            # shuffle_in_unison() is a method from P1 which shuffles two datasets randomly with a random state
            # shuffle_in_unison() put below load_data()
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=test_size,
                                                                                                    shuffle=shuffle)

    # cleans the data then returns it
    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result


def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


# ------------------------------------------------------------------------------
# Make predictions on test data
# ------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines
# x-PREDICTION_DAYS:x is used so that x_train is missing 60 days from its dataset which
# can be used as a timeframe for the model to predict. It appends the entire dataset 
# multiple times, removing one element until it finally reaches the end of the loop.

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
# ------------------------------------------------------------------------------
# Plot the test predictions
# To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
# ------------------------------------------------------------------------------

# plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
# plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
# plt.title(f"{COMPANY} Share Price")
# plt.xlabel("Time")
# plt.ylabel(f"{COMPANY} Share Price")
# plt.legend()
# plt.show()

# task 2
# candle stick charts
def display_candle_plots(df, start_date, end_date):
    # ensure df is a dataframe
    assert isinstance(df, pd.DataFrame), "df is not DataFrame"

    # validate start_date/end_date
    try:
        datetime.date.fromisoformat(start_date)
    except ValueError:
        raise ValueError("ds_start should be YYYY-MM-DD")
    try:
        datetime.date.fromisoformat(end_date)
    except ValueError:
        raise ValueError("ds_end should be YYYY-MM-DD")

    # creates a mask between the start and end date
    mask = (df.index > start_date) & (df.index <= end_date)
    # creates another dataframe that is only within the date range of the mask
    df2 = df.loc[mask]

    # attempt at loading SMA stuff, datasets downloaded through yahoo_fin don't contain it, seemingly
    # sma1 = fplt.make_addplot(task1test["test_df"]["SMA"], color="lime", width=1.5)
    # sma2 = fplt.make_addplot(task1test["test_df"]["SMA"], type="scatter", color="purple", marker="o", alpha="0.7", markersize=50)
    fplt.plot(df2, type="candle", title=f"Actual {COMPANY} Price from {start_date} to {end_date}", ylabel="$ Price", xlabel="Time")
    """
        draws a plot of specified type (ohlc, line, candle, renko, pnf)
        REQUIRES an argument of type DataFrame
        title can be specified
        label of Y-axis can be specified
        label of X-axis can be specified
        style of the plot can be specified ('binance', 'blueskies', 'brasil', 'charles', 'checkers', 
            'classic', 'default', 'ibd', 'kenan', 'mike', 'nightclouds', 'sas', 'starsandstripes', 'yahoo')
            can also be custom through the use of fplt.make_marketcolors() and fplt.make_mpf_style()
        multiple plots can be added to one display. create the addplots with fplt.make_addplot(), add them to the main plot
            with the addplot argument. typically used for technical indicators (SMA, EMA, RSI etc)
        volume of stocks traded in a day can be seen by setting the volume argument to True
            if volume is true, ylabel_lower can be used to give a label to it on the Y-axis
        plots can be saved with the savefig argument by passing in a filename. Ex "cba_2023_2024.png"
    """


def display_boxplot(df, start_date, end_date):
    # ensure df is a dataframe
    assert isinstance(df, pd.DataFrame), "df is not DataFrame"

    # validate start_date/end_date
    try:
        datetime.date.fromisoformat(start_date)
    except ValueError:
        raise ValueError("ds_start should be YYYY-MM-DD")
    try:
        datetime.date.fromisoformat(end_date)
    except ValueError:
        raise ValueError("ds_end should be YYYY-MM-DD")

    # creates a mask between the start and end date
    mask = (df.index > start_date) & (df.index <= end_date)
    # creates another dataframe that is only within the date range of the mask
    df2 = df.loc[mask]
    # df2 = df2.to_numpy()
    # print(df2)

    df2[["open", "high", "low", "adjclose"]].boxplot()
    # plt.boxplot(boxplot)
    plt.show()

    # have to display separately. if displayed with the other columns, squishes them too much
    df2[["volume"]].boxplot()
    plt.show()


task1test = load_data(ticker=COMPANY)
display_candle_plots(task1test["df"], "2024-04-01", "2024-06-01")
# display_boxplot(task1test["df"], "2024-04-01", "2024-06-01")

# ------------------------------------------------------------------------------
# Predict next day
# ------------------------------------------------------------------------------


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??
