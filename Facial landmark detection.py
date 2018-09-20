
import dlib
import cv2
import numpy as np
import imutils
from imutils import face_utils

def main():

    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    cap=cv2.VideoCapture(0)
        
    
    while True:

        ret,frame=cap.read()
        frame=imutils.resize(frame,width=500)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        rects=detector(gray,1)

        for (i,rect) in enumerate(rects):
            (x,y,w,h)=face_utils.rect_to_bb(rect)
            #cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),2)

            shape=predictor(gray,rect)
            shape=face_utils.shape_to_np(shape)
            
            for(x,y) in shape:
                cv2.circle(frame,(x,y),1,(255,255,255),-1)
                print(x,y)

        cv2.imshow("Output",frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cv2.destroyAllWindows()
    cap.release()
        


main()