# Generative Adversarial Network

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import Tensor

# Setting some hyperparameters
batchSize = 64 # We set the size of the batch.
imageSize = 64 # We set the size of the generated images (64x64).

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the generator
class G(nn.Module):
    # defining the arch
    def __init__(self):
        super(G, self).__init__() # For making the inheritance of the nn module class to the class G of object self to be used in the future
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100,512,4,1,0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2,1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128,4,2,1,bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,4,2,1,bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4,2,1, bias = False), #Here the output dimention corresponds to the image channels of the generator(RGB)
            nn.Tanh())
        # The main meta module contains different modules to indicate the different connections of the NN
        # We will create an inverse CNN here because it will take in a random Vector as input and give an image as output, inverse of that of the CNN
    def forward(self, input):
        output = self.main(input)
        return output
    # We now have to define the function for initializing the weights and to define the architecture of G

# Generator build
netG = G()
netG.apply(weights_init) # Applying teh weight init function to teh neural network

# Gnerating the Discriminator 
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128,256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256,512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512,1, 4, 1, 0, bias = False),
            nn.Sigmoid()
            )
    def forward(self,input): 
        output = self.main(input)
        return output.view(-1) # For flattening the 2D array of output obtained

# Creating the Disciminator
netD = D()
netD.apply(weights_init)

# Training the Brain of the DNGAN
criterion = nn.BCELoss()  
optimizerD = optim.Adam(netD.parameters(),lr =0.0002, betas= (0.5,0.999))
optimizerG = optim.Adam(netG.parameters(),lr =0.0002, betas= (0.5,0.999))

for epoch in range(25):
    for i, data in enumerate(dataloader,0):  
        # The data is analysed minibatch by minibatch
        
        # Updating the weights of the discriminator 
        netD.zero_grad() # Initializing the gradients of the weights to zero
        # Train the Discriminator with the fake image 
        real,_ = data
        inputs = Tensor(real)
        target = Tensor(torch.ones(inputs.size()[0])) # This is to specify that the ground truth is the real and is given by 1
        output = netD(inputs)
        errD_real = criterion(output, target) # For the real vs the fake images 
        # Training the Dsicrimitator with the fake images,hence to reject them
        noise = Tensor(torch.randn(inputs.size()[0], 100, 1, 1)) # We create a minibatch of vectors of size input.size and each vector having size 100
        fake = netG(noise)
        target = Tensor(torch.zeros(inputs.size()[0]))
        output = netD(fake.detach())
        err_fake = criterion(output, target)
        # Backpropagating the real and the fake error for training the Dsicriminator and the Generator
        errD = errD_real + err_fake
        errD.backward()
        optimizerD.step() # The step function applies the optimizer on the neural network of the discriminator to update the weights.
        
        # Updating the weights of the Generator 
        netG.zero_grad()
        # We need to train the generator against the target of one, as we need to make teh discriminator out close to one
        target = Tensor(torch.ones(inputs.size()[0]))
        output = netD(fake) # We don't detach the gradient because we need to update the weights 
        errG = criterion(output, target)
        errG.backward() # This constitutes the gradient based on which the backpropagation is occuring
        # Now the weights are updated as per the optimzer with which teh neural net was initialized 
        optimizerG.step()
        
        print('[%d/%d][%d/%d] Loss_D: %.4f , Loss_G: %.4f' % (epoch,25,i,len(dataloader), errD.data ,errG.data))
        if (i % 100 ==0):
            vutils.save_image(real, '%s/real_samples.png'%'./results', normalize= True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png'% ('./results', epoch), normalize= True)