#Data preprocessing 
# Importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from keras.utils import to_categorical
# importing the dataset 
dataset = pd.read_csv('Data.csv') 
x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1:].values


# Taking care of the missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

# We need to encode the categorical variables 
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
to_attach= []
to_attach = np.array(to_attach)
for i in range(pd.DataFrame.columns):
    if( type(x[0,i:i+1]) == str):   
        x[:,i:i+1] = labelencoder_x.fit_transform(x[:,i:i+1])
        z = to_categorical(x[:,i:i+1])
        to_attach = np.append(z,to_attach,axis =1)


# We are considering that the out put is having one feature.
labelencoder_y = LabelEncoder()
y[:,0]= labelencoder_y.fit_transform(y[:,0])


""" We must use the to_categorical class from the keras.utils library for making the onehotencoder, as the current onegotencoder fucntion is flawed 
and the particular one can we encoded as onehot by label encoding into integer encoded format and then doing the onehot on he seperate columns alone 
. This way, we can use the numpy function instead of the normal append that we do for list """

# Seperating the training set and the test set
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2 ,random_state = 0)

#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
""" We don't need to perform scaling on the dependent varaible because we have a classification problem 
with small categoric dependent variable"""