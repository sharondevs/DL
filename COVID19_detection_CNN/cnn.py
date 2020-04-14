# Building the CNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
# Initializing the CNN
classifier = Sequential()
# Adding the convolutional layer
classifier.add(Convolution2D(filters =32 , kernel_size = (3,3) , input_shape=(128,128,3), data_format ='channels_last', activation ='relu'))
# Pooling by MaxPooling 
classifier.add(MaxPooling2D(pool_size=(2,2), strides = 2))
# Adding other layers
classifier.add(Convolution2D(filters = 32, kernel_size = (3,3), data_format ='channels_last', activation ='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2), strides = 2))
classifier.add(Convolution2D(filters = 64, kernel_size = (3,3), data_format ='channels_last', activation ='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2), strides = 2))
classifier.add(Convolution2D(filters = 64, kernel_size = (3,3), data_format ='channels_last', activation ='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2), strides = 2))


# Flattening
classifier.add(Flatten())
# Fully connected layer at output layer
classifier.add(Dense(units = 64, activation = 'relu', kernel_initializer = 'uniform' ))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 64, activation = 'relu', kernel_initializer = 'uniform' ))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 1, activation = 'sigmoid')) # If the outcoome as categorical, then we have to use the softmax function 
# Compiling 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Preprocessing the image dataset, and generating more data by using keras to tackle overfitting due to low data
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
        target_size=(128,128),
        batch_size=32,
        class_mode='binary')

test_set = train_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128,128),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator( 
        training_set, # Training set
        steps_per_epoch=5400, # no, of samples thatwe should evaluate as one epoch, which is the no. of samples we have, right?
        epochs=1,
        validation_data=test_set,
        validation_steps= 620)

# We can also improve the accuracy of the train and test set by increasing either the convolutional layer
# or a fully connected layer. I added one more convolutional layer.
# Adding more layers, not only increases the accuracy of the training set, but also the accuracy of the test set. 
# Increasing the value of the pixel size we are taking also improves the accuracy
# For single prediction, we can use the predict function of the classfier object
# But for inputing the image into the model, we need to preprocess it so that it is made into the required form for the CNN
import numpy as np
from keras.preprocessing import image 
test_image = image.image.load_img('chest_xray/test/NORMAL/normal3.jpeg',target_size = (128,128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
pred = classifier.predict(test_image)
training_set.class_indices # For getting the class indices, i.e, the class of the outputs. This gives the category 
# to which the input image belongs to
# More accuracy can we obtained by using 4 convolution layers in total, we add 3 dense layers in total
# with dropout of 0.6/2 and again the same dropout

# The pre-trained model has loss: 0.1700 - accuracy: 0.9350 - val_loss: 0.4069 - val_accuracy: 0.8267
# 
# Saving the model 
model_json = classifier.to_json()
with open("<model_file_name>.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("<model_file_name>.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
"""
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
"""