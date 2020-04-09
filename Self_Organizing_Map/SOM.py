# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X =sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom # This is the class we need for the SOM as there is no inbuild libraries for SOM 
som = MiniSom(x = 10,y = 10,input_len = 15, sigma = 1.0, learning_rate = 0.5 )

# We needs to initialize the weights
som.random_weights_init(X)
som.train_random(data =X , num_iteration = 150)

# Visualizing the SOM neurons  
from pylab import bone, pcolor, colorbar,plot,show # We are gonna have to make the map from scratch, hence
# we declare the structure 

bone()# Structure grid
pcolor(som.distance_map().T) # We use this to calculate the MID for each winning neuron in the SOM.
colorbar()
# Here, we can see that the white colors correspond to the out-lying  winning nodes(fraudisters)
# We add marker to indicate the cutomers who cheated and got approved, apart from the customers who cheated 
# and didn't get approval
# Red circle who got didn't get approval, green square to correspond to customers how got approved
markers = ['o','s']
colors =['r', 'g']
# i gives the different customers and x gives the vector corresponding to each customer
for i,x in enumerate(X):
    # Winning node w
    w = som.winner(x)
    plot(w[0] + 0.5,w[1] + 0.5,markers[y[i]], markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',markersize = 10, markeredgewidth = 2) # w[0],w[1] gives the coordinates of the winning nodes on the som
    # Hence, we need to show the winning node marker on teh som, weuse the y data to select out of the markers
    # and add the plot to the som graph
    # when we are considering x from the winner function, we get the coordinates of that customer in the SOM
    # and the nthe plo function is used to plot the function teh color and marker
show()
# Now we need to catch the potential frauds who got approved but has very high distance
mapping = som.win_map(X) # This gives the mapping of the winning nodes on the SOM
# from this dictionary, we can obtain the datasets associated with each winning nodes 
# Now,we look at the map to identify the wining nodes, but having high MID and put the m together as frauds
# The coordinates of the tiles in the map are identified and the data of the customers in these tiles are attached into a single fraud list 
# from which the bank will look into deep and figure out whether if the customer has done anything wrong
frauds = np.concatenate( (mapping[(2,3)], mapping[(2,5)]), axis = 0 )
frauds = sc.inverse_transform(frauds)
