# Logistic Regression

#Data preprocessing 
# Importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from keras.utils import to_categorical
# importing the dataset 
dataset = pd.read_csv('Social_Network_Ads.csv') 
x = dataset.iloc[:,2:4].values
y = dataset.iloc[:,4].values

#The social network user would buy the incredibly priced SUV, based on teh salary and the age varaibles

# Seperating the training set and the test set
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25 ,random_state = 0)

#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
""" We don't need to perform scaling on the dependent varaible because we have a classification problem 
with small categoric dependent variable"""

# Here , we have a two dimentional dependency of input over the output 
# Fitting the logistic regression into the 2-D data 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)

# Predicting the test set results.
y_pred = classifier.predict(x_test)

# For comparing the results with the originla data,we have 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred )

# Visualization of the training test results 
from matplotlib.colors import ListedColormap
x_set, y_set = x_train , y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.figure(dpi=1400)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)): # This for loop is for plotting all the data points on the map
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, s=10, marker = 'o')
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# Now for testing,we have to plot x and y test dataset instead of x and y train set
x_set, y_set = x_test , y_test
plt.figure(dpi=1400)
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)): # This for loop is for plotting all the data points on the map
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, s=10, marker = 'o')
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()