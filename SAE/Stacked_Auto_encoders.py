# Autoencoder
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

# Building Stacked Autoencoder
# We need to declare a class for making the stacked autoencoder, that class should also inherit from the 
# nn.Module class imported form the nn module of torch
# We declare teh architecture of the Autoencoder
class SAE(nn.Module):
    def __init__(self,):
        super(SAE,self).__init__()
        self.fc1 = nn.Linear(nb_movies,20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    # Now we need to define a function to perform teh forward propagation of the dataset
    def forward(self, x):
        x= self.activation(self.fc1(x))
        x= self.activation(self.fc2(x))
        x= self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()  # This is for the loss function
optimizer = torch.optim.RMSprop(sae.parameters(), lr = 0.01,weight_decay =0.5) # The weight decay function is for preventing overfitting, to regulate convergence

# Training the autoencoder
nb_epoch = 200
for epoch in range(1,nb_epoch + 1):
    train_loss = 0
    s =0. # THIS INDICATE the number of users who have rated atleast one movie
    for id_users in range(nb_users):
        inputs = Variable(training_set[id_users]).unsqueeze(0)
        # The pytorch does not accept only one dimentional vector as input. 
        # Hence, we need to convert the input by adding one more dimention ,for the batch 
        target = inputs.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae.forward(inputs)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10 ) # This is needed because we are finding 
            # the loss for the movies which are rated by the user, hence we need to take care of the mean while calculating the loss during the 
            # computation
            # Now to consider the backward method, for updating the weights 
            loss.backward() # This actually decides which way the weights are to be updated, if the weights are to be increased or decreased
            train_loss+= np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step() # This actually applies the gradient decent algorithm and the amount by which the weights are updated
    print("Epoch: " + str(epoch)+ " loss: "+ str(train_loss/s))
# Now ,the loss function  that we get is indicating the error in the predicted output in-terms of stars
# If we gt a loss of 1, it indicated the error was made in output wrt the input by a start of 1

# Testing the SAE
test_loss = 0
s =0. # THIS INDICATE the number of users who have rated atleast one movie
for id_users in range(nb_users):
    inputs = Variable(training_set[id_users]).unsqueeze(0)
    # The pytorch does not accept only one dimentional vector as input. 
    # Hence, we need to convert the input by adding one more dimention ,for the batch 
    target = Variable(test_set[id_users]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae.forward(inputs)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10 ) # This is needed because we are finding 
        # the loss for the movies which are rated by the user, hence we need to take care of the mean while calculating the loss during the 
        # computation
        # Now to consider the backward method, for updating the weights 
        test_loss+= np.sqrt(loss.data*mean_corrector)
        s += 1.
print("test loss: "+ str(test_loss/s))
# The user input variable has to be the training_set because we need to predict the values of the ratings
# of the movies that the user have not watched yet and those values are ten compared with the test set results
# The trained data is fed into the model and the results are evaluated. Then the predicted ratings o the user for movies that they have not watched 
# gets compared with the ratings in the test set, to check how accuratly that the model have predicted the ratings that are in the testset
# Inference  : The u1 test set gave a test loss of 0.9505(less than 1 star) 
# Improvement is possible with adjusting the nodes in the hidden layers