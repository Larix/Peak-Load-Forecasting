# Peak-Load-Forecasting
Using some ways to predict peak load value.
### Autoregressive Integrated Moving Average model(ARIMA):
```
I try it to predict specific weekday (ex:trained by every Monday or Tuesday etc...)
```
![image](https://github.com/Larix/Peak-Load-Forecasting/blob/master/img/arima_rss.png)
![image](https://github.com/Larix/Peak-Load-Forecasting/blob/master/img/arima_rmse.png)


### LSTM:
```
I use lstm to predict long times data (trained by consecutive days)
```
![image](https://github.com/Larix/Peak-Load-Forecasting/blob/master/img/lstm_loss.png)
![image](https://github.com/Larix/Peak-Load-Forecasting/blob/master/img/lstm_prediction.png)
