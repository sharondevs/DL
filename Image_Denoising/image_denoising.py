## Denoising gray scale images using Auto Encoders 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as snb

""" The label translation is as follows:
    0 - T-Shirt
    1 - Trouser
    2 - Pullover
    3 - Dress
    4 - Coat
    5 - Sandal
    6 - Shirt
    7 - Sneaker
    8 - bag
    9 - Ankle boot
    
"""
## We load the dataset for the images to be obtained in grey scale
(X_train,y_train), (X_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()

# For sanity checking the image obtained,we plot the image
plt.imshow(X_train[0], cmap='gray')

## Now to perform the visualization 
i = random.randint(1,60000) # Produces a random number between 1 and 60000
plt.imshow(X_train[i], cmap = 'gray')
label = y_train[i]

# To do mroe visualization, we need to view the images in a grid of 15x15
W_grid = 10
L_grid = 10

fig , axes = plt.subplots(L_grid, W_grid, figsize = (17,17)) # This gives the subplot for the images
axes = axes.ravel() # This expands the 15x15 into 225 element list
n_training = len(X_train)

for i in np.arange(0, W_grid*L_grid):
    index = np.random.randint(0, n_training) # We initialize random interger for indexign out of the training set 
    axes[i].imshow(X_train[index]) # the axes array of objects of the subplot class can be instantitated with the images
    axes[i].set_title(y_train[index],fontsize = 8) # We set the title for the figures of the subplot on each figure
    axes[i].axis('off') 

## Data Preprocessing 
# We need to normaliza the data
X_train = X_train/255 # Due to the grey scale images
X_test = X_test/255

# Now, to add the noise

noise_factor = 0.3
noise_dataset = []

for img in X_train:
    noisy_image = img + noise_factor * np.random.randn(*img.shape)
    noisy_image = np.clip(noisy_image, 0,1)
    noise_dataset.append(noisy_image)
noise_dataset = np.array(noise_dataset)
# So to test the image obtained
plt.imshow(noise_dataset[22],cmap = 'gray')
# Now to add the noise to the testing data
noise_factor = 0.1
noise_test_dataset = []

for img in X_test:
    noisy_image = img + noise_factor * np.random.randn(*img.shape)
    noisy_image = np.clip(noisy_image, 0,1)
    noise_test_dataset.append(noisy_image)
noise_test_dataset = np.array(noise_test_dataset)

## Building the Autoencoder
autoencoder = tf.keras.Sequential()
# Encoder
autoencoder.add(tf.keras.layers.Conv2D(filters = 16, kernel_size=3,strides = 2,padding = 'same', input_shape=(28,28,1) ))
autoencoder.add(tf.keras.layers.Conv2D(filters = 8, kernel_size=3,strides = 2,padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2D(filters = 8, kernel_size=3,strides = 1,padding = 'same'))
# Decoder
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters = 16, kernel_size=3,strides = 2,padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size=3,strides = 2,padding = 'same', activation = 'sigmoid'))

autoencoder.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.Adam(lr=0.001))
autoencoder.summary()

## Now to fit the data for training the model
# I have trained the data for an epoch of 10
autoencoder.fit(noise_dataset.reshape(-1,28,28,1),
                X_train.reshape(-1,28,28,1),
                epochs = 100,
                batch_size = 200,
                validation_data = (noise_test_dataset.reshape(-1,28,28,1), X_test.reshape(-1,28,28,1)) )

## Evaluating the model
evaluation = autoencoder.evaluate(noise_test_dataset.reshape(-1,28,28,1),X_test.reshape(-1,28,28,1))
print('Test Accuracy : {:.3f}'.format(evaluation))

## Prediction of images to generate clear images
predicted = autoencoder.predict(noise_test_dataset[:10].reshape(-1,28,28,1))

## Visualization of the Generated predicted images
fig , axes = plt.subplots(2,10,sharex=True,sharey=True,figsize=(20,4))
for images,row in zip([noise_test_dataset[:10],predicted],axes):
    for img,ax in zip(images,row):
        ax.imshow(img.reshape((28,28)),cmap = 'Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        
        
                 
















