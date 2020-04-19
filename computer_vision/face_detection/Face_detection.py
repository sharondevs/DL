# Face Recognition 

# Importing the libraries
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Function for detection 
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(image =gray,scalaFactor = 1.3, minNeighbors = 5 )
    # This is to specify the width of the square of the face detection
    # The element face is a tuple that contain the upper left corner of the square and the width and height
    # Now we iterate through the images and inside these squares we detect the eyes
    # face consist of the coordinates of the rectangles that were detected by the face_cascade 
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y), (x+w,y+h), (0, 255, 0), thinkness = 2)
        # Now we look for the eyes, and for this we have the two region of interests , gray and frame
        roi_gray = gray[y:y + h, x:x+w] # Now we have the zone of interest for the face
        roi_color = frame[y:y + h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3 )
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh), (0, 0, 255), thinkness = 2)
    return frame # This gives the image on top of which the rectangles are marked 
# Now to configure the webcam
video_capture = cv2.VideoCapture(0)
while True:
    # We read tehe last frame coming from the camera and we need teh last element of the VideoCapture class 
    _, frame = video_capture.read()
    # Converting the RGB into black and while 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video',canvas)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break # This is make the detection stop if on pressing q
video_capture.release()
cv2.destroyAllWindows()

