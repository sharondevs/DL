## Single Shot Multibox Detection using Torch Tensors

This repo has a pre-trained SSD model based on pytorch. The object_detection.py is for detecting the object in the given video. 
This is acheived through the implementation of the pre-trained model and the SSD architecture, given in the repo. 
The model was trained on 20+ classes and is robust enough to be able to detect objects with more performance than any other object
detection algorithms. The given weights can we used to get very good results(have tested it), and can be applied for other video feeds.
This model can also be applied to other real world problems, because of its unique and robust nature.
Its advisable to run all the models on CUDA, and pytorch also supports CUDA acceleration.
The configurations of the image that needs to be input to the model is given in the objectdetection.py code 
which is in accordance with the image configurations with which the model was trained.
This shows that DNN can actually be very robust when used along with SSD, than other frameworks like opencv and algorithms like R-CNN and YOLO.
The video clips used for detection is attached along with the repository.

Datasets available at : http://host.robots.ox.ac.uk/pascal/VOC/
SSD architecture was taken from:
https://github.com/amdegroot/ssd.pytorch
(He has done a wonderful job! )
additional read : 
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
https://github.com/pytorch/tutorials/tree/master/beginner_source/blitz
https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd