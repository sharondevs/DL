# Object Detection 

# Importing the packages
import torch # It has dynamic graphs which help in effective computation of gradeint and back propagation
from torch.autograd import Variable # 
import cv2 # For drawing the rectangles around the detecter objects 
from data import BaseTransform, VOC_CLASSES as labelmap 
# Basetransform is for converting the input image to the required format, VOC_classes is a dictionary for encoding the classes
# text to integers mapping 
from ssd import build_ssd # This is for contructing the SSD 
import imageio # For processing  the images of the video and applying the detection function

#Definfing the detect function

def detect(frame, net, transform):
    height,width = frame.shape[0:2]
    # We need to first transform the frame to correcponding dimentions 
    frame_t = transform(frame)[0] # We only need the first returned element
    # Now to convert the frame_t array to torch tensors
    x = torch.from_numpy(frame_t).permute(2,0,1)# Changing the channel order
    # The SSD was trained on the GRB and hence we need to transform the x 
    # We need to add one more dimention to accept the input in batches 
    x = Variable(x.unsqueeze(0))# Added an additional dimension for the batch
    # We convert the torch tensor into tensor variable 
    
    # Now to fit the data into the pre-trained model
    y = net(x)
    detections = y.data 
    # we need to scale the positions of the detected objects
    # The variable y would be a four dimensional object with w,h,w,h for the opsitions of the upper left and the lower right corners 
    # of the rectangle
    scale = torch.Tensor((width,height,width ,height))
    # The detections tensor contains [batch_of_output for the corresponding input batches,number of classes of objects detected in input,
    # , number of occurance of the obejcts of each class, (score, x0,y0,x1,y1)]
    # The tuple of five elements gives the occurance for each object in a batch, and gives a score for it and the rest is the coordinates of the rectangular boxes
    # if the score is higher than 0.6, the we can say that an occurance has happened 
    # and we would get the coordinates of the detected occurance of teh objects 
    # Hence, we go for defining the occurance of the classes and conditionally defining to keep the classes for which the occurance is hgiher than 0.6
    for i in range(detections.size(1)): # Yeild the number of detectable classes
        j = 0
        while (detections[0,i,j,0] > 0.6 ): # Here i is the class of the object detected and j is the index of the occurance of the class i
            pt = (detections[0,i,j, 1:]*scale).numpy()  #We need to back convert torch tensors into numpy array bez opencv works with arrays
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])) , (int(pt[2]), int(pt[3])), (255,0,0), thickness = 2)
            cv2.putText(frame,labelmap[i-1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2,lineType= cv2.LINE_AA)#This gives the label of the ith class that we have 
            j +=1
    return frame

# Defining and import configuting the SSD NN
net = build_ssd('test') # building the NN
# The pre-trained model weights are imported 
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage,loc:storage ))
# the torch will open a tensor that will contain these weights and the the load_state_dict function is to attribute the laoded torch weights to the NN

# Now for the transform function for making the inputs compatible with the input frames
transform = BaseTransform(net.size, (104/256.0,117/256.0, 123/256.0))# This puts the called values to the right scale 
# Now detecting the objects on the video
reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4',fps=fps) # This is the output video maker
for i, frame in enumerate(reader):
    frame = detect(frame,net.eval(),transform)
    writer.append_data(frame)
    print(i)
writer.close()