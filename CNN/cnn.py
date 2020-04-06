# Building the CNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

# Initializing the CNN
classifier = Sequential()
# Adding the convolutional layer
classifier.add(Convolution2D(filters = 32, kernel_size = (3,3) , input_shape=(64,64,3), data_format ='channels_last', activation ='relu'))
# Pooling by MaxPooling 
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding another convolutional layer for more accuracy 
classifier.add(Convolution2D(filters = 32, kernel_size = (3,3 ), data_format ='channels_last', activation ='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Another two layers can be added
# Flattening
classifier.add(Flatten())
# Fully connected layer an output layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid')) # If the outcoome as categorical, then we have to use the softmax function 
# Compiling 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Preprocessing the image dataset, and generating mroe data by using keras to tackle overfitting due to low data
# This is follwed by fitting the generated data into the build CNN
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,  # Rescaling the pixels by dividing them with 255, thus making them between 0 and 1
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


classifier.fit_generator( 
        training_set, # Training set
        steps_per_epoch=8000, # no, of samples thatwe should evaluate as one epoch, which is the no. of samples we have, right?
        epochs=25,
        validation_data=test_set,
        validation_steps= 2000)
# We can also improve the accuracy of the train and test set by increasing either the convolutional layer
# or a fully connected layer. I added one more convolutional layer.
# Adding more layers, not only increases the accuracy of the training set, but also the accuracy of the test set. 
# Increasing the value of the pixel size we are taking also improves the accuracy
# For single prediction, we can use the predict function of the classfier object
# But for inputing the image into the model, we need to preprocess it so that it is made into the required form for the CNN
import numpy as np
from keras.preprocessing import image 
test_image = image.image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
pred = classifier.predict(test_image)
training_set.class_indices # For getting the class indices, i.e, the class of the outputs. This gives the category 
# to which the input image belongs to
# More accuracy can we obtained by using 4 convolution layers in total, we add 3 dense layers in total
# with dropout of 0.6/2 and again the same dropout
