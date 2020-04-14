## Disclaimer 
# This model cannot be considered as an accurate measure for detecting the symptoms for covid-19. 
The detection is trainied from the X-ray images of the covid-19. The dataset of the patients are 
collected from the open-source repo, attached along with the README. Again, I hold no liability 
for the correctness and accuracy of the model. This is a purely experimental attempt from my side, 
to show how deep learning technology can revolutionarize the way we detect disease in the future.

I was able to get a loss: 0.1700 - accuracy: 0.9350 , val_loss: 0.4069 - val_accuracy: 0.8267

The above results are solely-based on the dataset provided in the github repo and kaggle page(ref), 
which cannot be taken as a true form of data. Hence, this is just a proof of concept that Convolutional Neural
Network can be used for such rough estimations, provided the models are trained with large datasets and suffient accuracy
is acheived. The results shown are only based on 1 epoch(having a step of 5400 for training) and with a batch size of 32.
I believe we can attain more accuracy by tuning the hyper parameters even more(perhaps changing the batch size, feature map no., etc.)
I combined both of the datasets from the source and managed to get:
Training set : 5473 images belonging to 2 classes.
Test set : 624 images belonging to 2 classes.
The images can be placed in the respective folders inside the dataset main folder.

P.S : I was facing some lagginess while doing the training for CNN's(don't know why, have trained on both CPU and GPU, no difference...Probably some capping issue or somethings), hence i recommend to do more epochs, definely more than 1 ! 

Reference :
Datasets and additional read available on :
1) https://github.com/ieee8023/covid-chestxray-dataset
2) https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/data

 
 
