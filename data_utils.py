__author__ = 'Larix-Shen'

import sys, os
import time
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datetime import datetime
from statsmodels.tsa.stattools import adfuller

def load_raw_data_to_process():
    df = pd.read_csv('data\\raw_data.csv')  
    # pre-processing
    df = df.iloc[1:, :7]
    df.rename(columns={df.columns[0]: "date", df.columns[1]: "clear_peak", df.columns[2]: "peak", 
                        df.columns[3]: "operating_reserve", df.columns[4]: "p_operating_reserve", 
                        df.columns[5]: "industry_usage", df.columns[6]: "house_usage"}, inplace = True)

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.set_index('date')
    df['weekday'] = pd.Series(df.index.dayofweek, index=df.index)
    weekday_df = convert_to_weekday_form(df, 'peak')

    #show_data_fig(df, weekday_df)
    return df, weekday_df

def smoothing(weekday_df):
    ts_log = np.log(weekday_df)

    moving_avg = ts_log.rolling(4).mean()
    ts_log_diff = ts_log - ts_log.shift(periods=1)
    ts_log_diff.dropna(inplace=True)
    
    ### check stationarity ###
    #plt.plot(ts_log.peak_Mon)
    #plt.plot(moving_avg.peak_Mon, color='red')
    #plt.show()
    #plt.plot(ts_log_diff.peak_Mon)
    #plt.show()
    #test_stationarity(ts_log_diff.peak_Mon)
    return ts_log, ts_log_diff

def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(4).mean()
    rolstd = timeseries.rolling(4).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

def get_train_test_set():
    load_raw_data_to_process = load_raw_data_to_process()
    train = df.sample(frac=0.8, random_state=200) #random state is a seed value
    test = df.drop(train.index)

# https://github.com/advaitsave/Introduction-to-Time-Series-forecasting-Python/blob/master/Time%20Series%20in%20Python.ipynb
def show_data_fig(df, weekday_df):
    show_specific_day_figure(df, 0)
    show_weekday_figure(weekday_df)
    show_peak_boxplot(weekday_df)

def show_peak_boxplot(weekday_df):
    #df.boxplot(column=list(df.columns))
    sns.boxplot(data=weekday_df)
    plt.show()

def show_weekday_figure(weekday_df):
    weekday_df.plot(title='Peak(MW)', figsize=(15, 6))
    plt.show()   

def show_specific_day_figure(df, day_number):
    df = df.loc[df['weekday'] == day_number, ['peak']]
    df.plot(title='Peak(MW)', figsize=(15, 6))
    plt.show()   

def convert_to_weekday_form(df, column_name):
    weekday_data = {}
    days = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
    for i in range(7):
        #week_df = df.loc[df['weekday'] == i, [column_name]]
        weekday_data[column_name + "_" + str(days[i])] = df.loc[df['weekday'] == i][column_name].tolist()
    weekday_df = pd.DataFrame.from_dict(weekday_data)
    weekday_df.index.name = 'week'
    return (weekday_df)

#ref:https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623
def detect_outliers_data(timeseries):
	print(find_anomalies(timeseries))

def find_anomalies(random_data):
    # Set upper and lower limit to 3 standard deviation
    anomalies = []
    random_data_std = np.std(random_data)
    random_data_mean = np.mean(random_data)
    anomaly_cut_off = random_data_std * 3
    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    print(lower_limit)
    print(upper_limit)
    # Generate outliers
    for outlier in random_data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies