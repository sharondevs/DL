## Training the SSD for object detection using Deep Learning

It is recommmended to run the training on CUDA enabled computers as the process is very performance intensive.
Hence, its a necessity that we need an NVIDIA powered CUDA enalbed GPU for processing with torch tensors.
This reduces the training time considerably. 
The VOC datasets are given in the 'VOCdevkit' in the data folder. The datasets can be downloaded from the VOC pascal website 
or by using the .sh files provided in the scripts folder.
The SSD model is initialized with the .pth file given in the weights folder.

dataset :
http://host.robots.ox.ac.uk/pascal/VOC/

additional read :
https://github.com/amdegroot/ssd.pytorch
