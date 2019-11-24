import sys, os
import time
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime

df = pd.read_csv('data\\raw_data.csv')  


days = np.array(df.iloc[0:, 0])
clear_peak = np.array(df.iloc[0:, 1]).astype(np.float)
peak = np.array(df.iloc[0:, 2]).astype(np.float)
operating_reserve = np.array(df.iloc[0:, 3]).astype(np.float)
p_operating_reserve = np.array(df.iloc[0:, 4]).astype(np.float)
industry_useage = np.array(df.iloc[0:, 5]).astype(np.float)
house_useage = np.array(df.iloc[0:, 6]).astype(np.float)


dates = [datetime.strptime(format(day), '%Y%m%d').date() for day in days]

'''

fig_plot = plt.figure(figsize=(12,4),dpi=120)
ax1 = fig_plot.add_subplot(1,1,1)
ax1.plot(dates, peak, lw=1, c='g')
ax1.set_title('peak (MW)')
plt.show()
'''
print(np.median(peak))

#ref:https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623
def detect_outliers_data():
	print(find_anomalies(peak))

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

if __name__ == '__main__':
	detect_outliers_data()
