import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_utils import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def normalize(train):
	train = train.drop(["Date"], axis=1)
	train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

	return train_norm

def build_train_for_scaler(train, past=30, future=5):
	X_train, Y_train = [], []
	for i in range(train.shape[0]-future-past):
		X_train.append(np.array(train[i:i+past]))
		Y_train.append(np.array([data[1] for data in train[i+past:i+past+future]]))

	return np.array(X_train), np.array(Y_train)

def shuffle(X,Y):
	np.random.seed(10)
	randomList = np.arange(X.shape[0])
	np.random.shuffle(randomList)
	return X[randomList], Y[randomList]

def splitData(X,Y,rate):
	X_train = X[int(X.shape[0]*rate):]
	Y_train = Y[int(Y.shape[0]*rate):]
	X_val = X[:int(X.shape[0]*rate)]
	Y_val = Y[:int(Y.shape[0]*rate)]
	return X_train, Y_train, X_val, Y_val

def buildOneToOneModel(shape):
	model = Sequential()
	model.add(LSTM(50, input_length=shape[1], input_dim=shape[2], return_sequences=True))
	# output shape: (1, 1)
	model.add(TimeDistributed(Dense(1)))		# or use model.add(Dense(1))
	model.compile(loss="mse", optimizer="adam")
	model.summary()
	return model


df, weekday_df = load_raw_data_to_process()
weekday_df_log, weekday_df_log_diff = smoothing(weekday_df)

min_max_scaler = MinMaxScaler()
train_norm = min_max_scaler.fit_transform(df)

X_train, Y_train = build_train_for_scaler(train_norm, 1, 1)
X_train, Y_train = shuffle(X_train, Y_train)
X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

# from 2 dimmension to 3 dimension
Y_train = Y_train[:,:,np.newaxis]
Y_val = Y_val[:,:,np.newaxis]

model = buildOneToOneModel(X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
history = model.fit(X_train, Y_train, epochs=200, batch_size=64, validation_data=(X_val, Y_val), callbacks=[callback])

y_pred = model.predict(X_val)
y_pred = y_pred.flatten()
error = mean_squared_error(Y_val.reshape(-1, 1), y_pred.reshape(-1, 1))

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print("Error is", error, y_pred.shape, Y_val.shape)
print(y_pred[0:15])
print(Y_val[0:15])

#https://de-yu-code.blogspot.com/2018/03/sklearnpreprocessingminmaxscaler.html
#https://www.cnblogs.com/chaosimple/p/4153167.html
y_pred = y_pred.reshape(-1, 1)
Y_val = Y_val.reshape(-1, 1)
y_pred_org = (y_pred * min_max_scaler.data_range_[1]) + min_max_scaler.data_min_[1] # min_max_scaler.inverse_transform(y_pred)
y_test_t_org = (Y_val * min_max_scaler.data_range_[1]) + min_max_scaler.data_min_[1] # min_max_scaler.inverse_transform(y_test_t)

# Visualize the prediction
from matplotlib import pyplot as plt
plt.figure()
plt.plot(y_pred_org)
plt.plot(y_test_t_org)
plt.title('Prediction vs Real Peak')
plt.ylabel('Peak(MW)')
plt.xlabel('Days')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.show()
