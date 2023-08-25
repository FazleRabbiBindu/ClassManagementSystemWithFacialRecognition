import cv2 as cv

capture = cv.VideoCapture(0) #video camera capture.0 stadns for webcam

tracker = cv.TrackerCSRT_create() #generates object tracker. this is a CSRT tracker 

success, img = capture.read() #reads image from video input

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1)

box = cv.selectROI("Tracker", img, False) #generates a selection area instance
tracker.init(img,box) #bounds box with image

def drawBox(img,box):
    x,y,w,h = int(box[0]),int(box[1]),int(box[2]),int(box[3])
    cv.rectangle(img,(x,y),((x+w),(y+h)),(250,0,250),3,1)


while True:
    success, img = capture.read()
    success, box = tracker.update(img)
    if success:
        drawBox(img,box)
    cv.imshow('camera',img)
    if cv.waitKey(1) & 0xff == ord('q'):
        break