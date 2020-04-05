## NOTES
# This script is compatible with python 3.5.5, hence please create and environemnt and run the project.
# The command for creatng the environment is : conda create <env_name> python=3.5.5 anaconda
# After creation, go into the environment and install keras and tensorflow. Theano can also be pip installed. Make sure the tensorflow installed is tensorflow-gpu

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variables of columns 1 and 2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X= X[:,1:] # This is for avoiding the dummy variable trap, hence no need for one dummy variable ,as they are dependent on each other.
# The dependent varaible is already in the integer encoded, hence no need for encoding twice


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# For training the model
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Classifier model
classifier = Sequential()
# Now, we have imported the layer , and then we start adding the layers one my one, starting with the input layer and then hidden layers.
# Followed by the output layer

#Adding the input and the hidden layers with dropout
classifier.add(Dense(units = 6,activation= 'relu', kernel_initializer= 'uniform',input_shape=(11,) )) # Specifying the first hidden layer, automatically gives the input layer, by adding the input contraints in the 'input_shape' arg
classifier.add(Dropout(rate=0.1)) # We apply dropout to the hidden layer
# No need for the input arg in the upcoming layers. 
# Adding the second hidden layer
classifier.add(Dense(units = 6,activation= 'relu', kernel_initializer= 'uniform'))
classifier.add(Dropout(rate=0.1)) # We apply dropout to the hidden layer
# Now we add the output layer 
classifier.add(Dense(units = 1,activation= 'sigmoid', kernel_initializer= 'uniform'))
# Now compiling the ANN 
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
# Now for fitting the train data into ANN for interconenction of the layers
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Now to convert y_pred into true or false to compare 
y_pred = (y_pred>0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Now for comparing if the predicted accuracy is true 
acc = (cm[0,0]+ cm[1,1])/(pd.DataFrame(y_test).shape)[0]
print(acc) # Thsi gives the accuracy
# Now lets try out our own test set 

#For inputing single values for prediction
test = []
test = np.array(test)
print("Enter the data correcponding to the option no: \n  1 : Credit Score \n 2 : Geography \n 3 : Gender \n 4 : Age \n 5 : Tenure \n 6 : Balance \n 7 : No of Products \n 8: Credit card \n 9 : Is active member \n 10 : Estimated Salary ")
for i in range(0,np.shape(X)[1]):
    print("\n Enter for option ",i,":")
    value = input()
    test =np.append(test,value)
X = dataset.iloc[:, 3:13].values
X = np.append(X,[test], axis =0)
# To categorize 
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X= X[:,1:] 
# Now to scale 
scale = StandardScaler()
X= scale.fit_transform(X)
x_needed= X[1000,:]
y_result = classifier.predict(np.array([x_needed]))
y_result = (y_result>0.5)

#To improve the model, we are using the k-fold cross validation 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6,activation= 'relu', kernel_initializer= 'uniform',input_shape=(11,) )) # Specifying the first hidden layer, automatically gives the input layer, by adding the input contraints in the 'input_shape' arg
    classifier.add(Dense(units = 6,activation= 'relu', kernel_initializer= 'uniform'))
    classifier.add(Dense(units = 1,activation= 'sigmoid', kernel_initializer= 'uniform'))
    classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
    return classifier
# Now we declare a global classifier 
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10)
accuracies = cross_val_score(estimator = classifier, X= X_train, y=y_train, cv = 10, fit_params={'epochs':100})  
# I cannot specify the epochs in the classifier object, i.e, the KerasClassifier object, hence we need to specify it in the cross_val_score,directly to the epoch defining function to prevent it from 
# going default
mean = accuracies.mean()
variance = accuracies.std() # The variance gives us the estimate whether if the data is overfitted into the model or not 
# Incase of low variance, we can say that the model is accurate and overfitting have not happened
#Incase overfitting happens, we have the drop out regularization technique for solving it

# We also tune the hyper parameters
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6,activation= 'relu', kernel_initializer= 'uniform',input_shape=(11,) )) # Specifying the first hidden layer, automatically gives the input layer, by adding the input contraints in the 'input_shape' arg
    classifier.add(Dense(units = 6,activation= 'relu', kernel_initializer= 'uniform'))
    classifier.add(Dense(units = 1,activation= 'sigmoid', kernel_initializer= 'uniform'))
    classifier.compile(optimizer= optimizer, loss= 'binary_crossentropy', metrics= ['accuracy'])
    return classifier
# Now we declare a global classifier 
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,32],'epochs':[100,500],'optimizer':['adam','rmsprop']} # We gvie different types of optimizers, epoch size and batch size 
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy', cv=10 )
grid_search = grid_search.fit(X_train,y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
