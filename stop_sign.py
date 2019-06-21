import cv2
import numpy as np
import glob

capture = cv2.VideoCapture(0)
capture.set(3, 640)#width
capture.set(4, 480)#height


stop_cascade = cv2.CascadeClassifier("stop_sign.xml")

while True:
    okay, image = capture.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detection
    cascade_obj = stop_cascade.detectMultiScale(gray,
                                                      scaleFactor=1.1,
                                                      minNeighbors=5,
                                                      minSize=(30, 30),
                                                      flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                                                      )

    # draw a rectangle around the objects
    for (x_pos, y_pos, width, height) in cascade_obj:
        cv2.rectangle(image, (x_pos+5, y_pos+5), (x_pos+width-5, y_pos+height-5), (255, 255, 255), 2)
        # stop sign
        if width/height == 1:
            cv2.putText(image, 'STOP', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Frame", image)
    
    key = cv2.waitKey(33)
    if key == 27:
        break

    
capture.release()
cv2.destroyAllWindows() 






    
