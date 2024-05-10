import cv2, numpy, os


cam=cv2.VideoCapture("cars\people.mp4")
facexml=cv2.CascadeClassifier("cars\haarcascade_frontalface_default.xml")
eyexml=cv2.CascadeClassifier("cars\haarcascade_eye.xml")


width,height=130,100

count=1

while count>0:
    ret, frame=cam.read()
    
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)



    face=facexml.detectMultiScale(grey,3,6)
    print(face)

    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)


    eye=eyexml.detectMultiScale(grey,3,6)
    print(eye)

    for (x,y,w,h) in eye:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)



    cv2.imshow("webcam",frame)
    k=cv2.waitKey(10)
    if k==27:
        break

