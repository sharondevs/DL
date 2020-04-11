#   Restricted Boltzmann Machine 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
# Here, the seperators is not comma, but the above symbol.
# The moveis.dat file does not have a column names
# We must specify special encoders to encode the movies as the  movies may contain special characters
# We would not be using the movie title in the BM, and instead the movie id given in the first column
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
# The last column is the zip code corresponding to the user
# The third column is code corresponding to the users job 
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
# first column is the user id, second column is the movies id
# the tthird column is the rating of that user, fourth column is ignores(timestamp)
# Now to cleare the training and the test set

# For preparing the data, we take only one base(training set) and test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t') # Here the seperator is a tab
# 80% train-test split
# Column 1 coresponds to the user, 2 to the movies id , 3rd is the rating and fourth is the timestamp
training_set = np.array(training_set , dtype = 'int')
# For test set
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set , dtype = 'int') # The test spilit is 20% 
# Total number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# We need to contruct a matrix having the columns as movies and the rows as user id, and with ratings as the cell value 
# We need this to be fed into the Boltzmann Machine 
def convert(data):
    new_data =[]
    for id_users in range(1, nb_users +1):
        # Now we need to get the movie id's of the user corrsponding to the id specified by the id_users,so that, we have 
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the lists into torch tensors
# tensors they are multidimentional structures, that are having elements of only one datatype
training_set = torch.FloatTensor(training_set) # This only accepts list-of-lists as arguments 
test_set = torch.FloatTensor(test_set)

# Building the RBM

# Converting the ratings into binary outputs 1- Liked, 0- Not Liked, by the user
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set == 3] = 1
training_set[training_set == 4] = 1
training_set[training_set == 5] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Building the architecture for RBM
# Building the class for RBM
# We need to input values like the no. of visible nodes and the no. of hidden nodes 
# This we input this along with a self function 
class RBM(): 
    def __init__(self,nv,nh):
        self.W =  torch.randn(nh,nv) # This produces a tensor object with size nh,nv having random initialized values of mean 0 and variance 1
        self.a = torch.randn(1,nh) # This produces the bias for the probablities of the visible nodes nv 
        # in accordance to the hidden nodes
        # but the bias should be 2D, having the shape (batch_size, the randn bias)
        self.b = torch.rand(1, nv)
        # Making the function that samples the hidden nodes
    def sample_h(self, x): # Where x corresponds to the no. of visible nodes when computing for the hidden nodes
        # Now,according to the formula, we need to compute the product of the weights vector and the vsible probablities x and then
        # add it along with the bias and pass it through the activation function(sigmoid)
        # Now this corresponds to a particular hidden node characteristics and helps in defining when the hidden nodes should react
        wx = torch.mm(x,self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v =  torch.sigmoid(activation) # This gives the probablities of the hidden node being activates given the visible nodes
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y): # Now this is for estimating the probablities of the visible node is activates(1), given the hidden node probablities
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h =  torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0,vk,ph0,phk):  # This is to train the visible node vectors and the probablities of the hidden node being activated, given the vidible node vector
        self.W += torch.sum((torch.mm(v0.T, ph0) - torch.mm(vk.T, phk)) +0) # A small problem in this spet
        self.b += torch.sum((v0-vk), 0)
        self.a += torch.sum((ph0-phk), 0)
        
# Importing the class RBM
# We need nv and nh. Hence, we have the no. of visible nodes as the no. of movies that we are feeding in 
# And for the nh, we can choose a relevent number. nh corresponds to the number of features that the movies have
# Thus, we choose the relevent number of features that we should take into account 
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv,nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1,nb_epoch + 1):
    # We introduce the loss variable 
    train_loss = 0
    # We need to normalize the train loss, we divide the train_loss with a counter
    s = 0. # Float counter
    for id_users in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_users : id_users + batch_size ]
        v0 = training_set[id_users : id_users + batch_size ]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print("Epoch: "+ str(epoch)+ " "+ "Loss: "+ str(train_loss/s))
    
# Testing the RBM

# We need to do the prediction according to the Markov chain Monticarlo technique
# We introduce the loss variable 
test_loss = 0.
# We need to normalize the train loss, we divide the train_loss with a counter
s = 0. # Float counter
for id_users in range(nb_users):
    v = training_set[id_users : id_users + 1]
    vt = test_set[id_users : id_users + 1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print("Test Loss: "+ str(test_loss/s))
# Hence, from the loss percent, we are able to tell that the model has managed to produce the results
# and have obtained sufficient accuracy. We can't visualize any results, but can see that the results are there....
# 