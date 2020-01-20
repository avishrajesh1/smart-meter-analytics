import pandas as pd
#si=['da','date','val']
#df = pd.read_excel(r"modified data.xlsx",columns=['da','date','val'])
import pandas as pd
import numpy
#import 
from numpy import nan
import matplotlib
from matplotlib import pyplot

df=pd.read_csv(r"C:\Users\Avish Khosla\OneDrive\Documents\household_power_consumption.txt",sep=';',header=0,low_memory=False,infer_datetime_format=True, parse_dates={'datetime':[0,1]},index_col=['datetime'])
print(df)
df.replace('?',nan,inplace=True)
df=df[['Global_active_power']].astype('float16')
df=df.resample("30Min").mean()

#df=df.set_index(df['DATE'])
#print(df)
#df4= df[df.index.weekday==7]
#df3= df[df.index.weekday==6]

df['prehr_1']=df['Global_active_power'].shift(1)
df['prehr_2']=df['Global_active_power'].shift(2)
df['prehr_3']=df['Global_active_power'].shift(3)
df['preday_1']=df['Global_active_power'].shift(48)
df['preday_2']=df['Global_active_power'].shift(48*2)
df['preday_3']=df['Global_active_power'].shift(48*3)
df['preweek_1']=df['Global_active_power'].shift(48*7)
df['preweek_2']=df['Global_active_power'].shift(48*7*2)
df['premo_1']=df['Global_active_power'].shift(48*30)
df['val']=df['Global_active_power']
df=df.drop(columns=['Global_active_power'])
df.dropna(inplace=True)


import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	dataX=dataset[:,0:look_back]
	dataY=dataset[:,look_back]
	'''
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
		'''
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility

dataset = df.values
print(dataset)
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 9

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1],1))
# create and fit the LSTM network
print(trainX)
model = Sequential()
model.add(LSTM(4, input_shape=( testX.shape[1],1)))
model.add(Dense(1)) 
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
'''
newtrain=trainX+trainPredict
print(newtrain)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
'''
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
print(trainPredictPlot.shape)
print(trainPredict.shape)
trainPredictPlot[0:train_size, look_back-1:look_back] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[train_size:len(dataset),  look_back-1:look_back] = testPredict
# plot baseline and predictions
plt.plot(df['val'])
df['pre']=trainPredictPlot[:,look_back-1:look_back]
df['pro']=testPredictPlot[:,look_back-1:look_back]
plt.plot(df['pre'])
plt.plot(df['pro'])
plt.show()
#df4['val2']=df2['Global_active_power']
print(df)
