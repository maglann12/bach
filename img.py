import cv2
cap=cv2.VideoCapture(0)
while True:
     ret,frame=cap.read()
     frame=cv2.resize(frame,(640,480))
     
     cv2.imshow('FRAME',frame)
     if cv2.waitKey(1) == ord('a'):
        print ("pressed a")
        frame=cv2.imwrite("/home/pi/opencv-distance/rf.png",frame)
        break
        
cap.release()
cv2.destroyAllWindows()


