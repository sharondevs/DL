# Importing the libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
# Importing the training set
dataset_training = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_training.iloc[:,1:2].values # This makes the training set for the NN, with the stock price.
# Now we scale the dataset by normalized scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
# For processing a dataset with 60 timesteps and one output
X_train =[]
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i,0]) # For getting the 60 observations from 0 to 59, i the first iteration
    y_train.append(training_set_scaled[i,0]) # For getting the 60th observation , in the first iteration
X_train,y_train = np.array(X_train),np.array(y_train)
# For changing the shape of the X_train for inputing into the network
X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1 ))

# Build
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers  import Dropout

regressor = Sequential()
# Now we add the LSTM layer. For that, we have to first speciify the number of LSTM units in the layer 
# Corresponding to the number of neurons in each layer. This increases the dimentionality, hence improving prediction
# We choose 50 neurons
regressor.add(LSTM(units = 50, return_sequences = True,input_shape = (X_train.shape[1], 1))) # We are only inputing the shape of the input sequence as the timestamp and the no.of dimentionalities/indicators
regressor.add(Dropout(rate =0.2))
# Addign three more LSTM layer to increase complexity with dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate =0.2))
regressor.add(LSTM(units = 50, return_sequences = True)) # the return_sequence is kept true because we need the output from the LSTM layer
regressor.add(Dropout(rate =0.2))
regressor.add(LSTM(units = 50, return_sequences = False)) # Since we are done adding the LSTM layers
regressor.add(Dropout(rate =0.2))
# Adding the output layer
regressor.add(Dense(units = 1)) # The no.of neurons in the output layer is one because the output is a one dimentional real number, hence one need one neuron to 
# give the output
# Compiling the model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') # instead of the RMSprop, we still use adam because it is very powerfull and usefull for RNN
# The loss function is the mean of the squareed difference between the predicted and the original values 
# Fitting the dataset into the RNN
regressor.fit(X_train,y_train, epochs = 100, batch_size = 32)

# We need to import the real google stock chart, for the year 2017, which we are supposed to predict by the model
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# For prediction , we concatenate the training and the test set
dataset_total = pd.concat((dataset_training['Open'],dataset_test['Open']),axis = 0)
# Now, we need to get the dataset_values ,which are the 60 timesteps before each predicted result. 
# Hence, we import only the stock observations that we need from the above made dataset
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60 :].values
inputs = inputs.reshape(-1,1) # This produces the single column array needed.

# Scaling the input
inputs = sc.transform(inputs) # Because the training set was fitted with the same object, hence
# the input should also be fitted with the same scale that is used by the training set for the RNN  
X_test =[]
for i in range(60, 80):
    X_test.append(inputs[i-60:i,0]) # For getting the 60 observations from 0 to 59, i the first iteration
X_test = np.array(X_test)
# For changing the shape of the X_train for inputing into the network
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1 ))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the predicted stock 
plt.figure(dpi=1400)
plt.plot(real_stock_price, color ='red', label ='Real Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
plt.title('Google Stock Chart prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Evaluating the model by Root mean square error between the predicted as well as the real stock prices
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

# Improvement of RNN
"""
Getting more training data: we trained our model on the past 5 years of the Google Stock Price but it would be even better to train it on the past 10 years.
Increasing the number of timesteps: the model remembered the stock prices from the 60 previous financial days to predict the stock price of the next day. That’s because we chose a number of 60 timesteps (3 months). You could try to increase the number of timesteps, by choosing for example 120 timesteps (6 months).
Adding some other indicators: if you have the financial instinct that the stock price of some other companies might be correlated to the one of Google, you could add this other stock price as a new indicator in the training data.
Adding more LSTM layers: we built a RNN with four LSTM layers but you could try with even more.
Adding more neurones in the LSTM layers: we highlighted the fact that we needed a high number of neurones in the LSTM layers to respond better to the complexity of the problem and we chose to include 50 neurones in each of our 4 LSTM layers. You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.
"""