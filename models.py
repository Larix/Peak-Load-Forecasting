from data_utils import *

def auto_regression(ts_log_diff):
    from statsmodels.tsa.ar_model import AR
    from random import random
    model = AR(ts_log_diff)
    model_fit = model.fit()


    plt.plot(ts_log_diff)
    plt.plot(model_fit.fittedvalues, color='red')
    plt.title('RSS: %.4f' % np.nansum((model_fit.fittedvalues - ts_log_diff) ** 2))
    plt.show()
    
    return model_fit

# https://machinelearningmastery.com/make-sample-forecasts-arima-python/


def auto_arima_moving_average(y):
	from pmdarima.arima import auto_arima
	# fit model
	train = y[:int(0.75*(len(y)))]
	valid = y[int(0.75*(len(y))):]

	model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)

	model_fit = model.fit(train)
	forecast = model.predict(n_periods=len(valid))
	forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])

	plt.plot(y, label='Train')
	#plt.plot(valid, label='Valid')
	plt.plot(forecast, label='Prediction')
	plt.show()
	#print(model_fit.summary())

	return model_fit

def ari_moving_average(ts_log_diff):
	from statsmodels.tsa.arima_model import ARIMA
	from pmdarima.arima import auto_arima
	# fit model
	model = ARIMA(ts_log_diff, order=(7,0,1))

	model_fit = model.fit(disp=False)
	plt.plot(ts_log_diff)
	plt.plot(model_fit.fittedvalues, color='red')
	plt.title('RSS: %.4f'% np.nansum((model_fit.fittedvalues-ts_log_diff)**2))
	plt.show()
	forecast = model_fit.forecast()[0]
	print(forecast)
	#print(model_fit.summary())

	return model_fit

def reverse_transformations(model, ts_log, y, weekday_df_log_diff):
	predictions_ARIMA_diff = pd.Series(model.fittedvalues, copy=True).append(pd.Series(model.forecast(steps=7)[0]), ignore_index=True)
	print(predictions_ARIMA_diff)
	predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

	predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
	predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
	# Taking Exponent to reverse Log Transform
	predictions_ARIMA = np.exp(predictions_ARIMA_log)
	RMSE = np.sqrt(np.nansum((predictions_ARIMA-y.peak_Wed)**2)/len(y.peak_Wed))
	predictions_ARIMA += np.sqrt(np.nansum((predictions_ARIMA-y.peak_Wed)**2)/len(y.peak_Wed))
	#print(predictions_ARIMA[:30][:])
	plt.plot(y.peak_Wed)
	plt.plot(predictions_ARIMA)
	plt.title('RMSE: %.4f'% np.sqrt(np.nansum((predictions_ARIMA-y.peak_Wed)**2)/len(y.peak_Wed)))
	plt.show()

if __name__ == '__main__':
    df, weekday_df = load_raw_data_to_process()
    '''
    df_log, df_log_diff = smoothing(df)
    model = auto_regression(df_log_diff.peak)
    reverse_transformations(model, df_log.peak, df, df_log_diff.peak[12:])
    '''
    weekday_df_log, weekday_df_log_diff = smoothing(weekday_df)
    #ar = auto_regression(weekday_df_log_diff.peak_Mon*-1)
    arma = ari_moving_average(weekday_df_log_diff.peak_Mon[:-6]*-1)
    reverse_transformations(arma, weekday_df_log.peak_Mon, weekday_df, weekday_df_log_diff)

    #auto_regression(weekday_df_log_diff.peak_Mon)
    #auto_regression(weekday_df_log_diff.peak_Tue)
    #auto_regression(weekday_df_log_diff.peak_Wed)
    #auto_regression(weekday_df_log_diff.peak_Thu)
    #auto_regression(weekday_df_log_diff.peak_Fri)
    #auto_regression(weekday_df_log_diff.peak_Sat)
    #auto_regression(weekday_df_log_diff.peak_Sun)