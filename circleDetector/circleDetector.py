import numpy as np
import cv2


#img=cv2.imread('circledetector4.jpg')
#print(img.shape)
cap=cv2.VideoCapture('circledetector.mp4')
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('circleoutput.mkv', fourcc, 20.0,  (frame_height, frame_width))
#frame=cv2.resize(img,(225,400))
while cap.isOpened():
    ret,frame=cap.read()
    if ret==True:
        
        blur = cv2.GaussianBlur(frame,(3,3),0)

        cv2.imshow('circle',blur)
        #hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        

        l_b = np.array([0,40,30])
        u_b = np.array([40,160,110])

        mask = cv2.inRange(frame, l_b, u_b)
        kernal=np.ones((1,1),np.uint8)

        mask=cv2.dilate(mask,kernal,iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        (contour, _) = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contour:
            area=cv2.contourArea(cnt)
            approx=cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            x=approx.ravel()[0]
            y=approx.ravel()[1]
            if area>400:
                cv2.drawContours(frame,[approx],0,(0,0,0))

                if len(approx) == 3:
                    cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                elif len(approx) == 4:
                    x1 ,y1, w, h = cv2.boundingRect(approx)
                    aspectRatio = float(w)/h
                    print(aspectRatio)
                    if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                        cv2.putText(frame, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    else:
                        cv2.putText(frame, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                elif len(approx) == 5:
                    cv2.putText(frame, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                elif len(approx)==6:
                    cv2.putText(frame, "Hexagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                else: 
                    cv2.putText(frame, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))




        #cv2.imshow("mask", mask)
        #cv2.imshow("res", res)
        
        cv2.imshow('frame',frame)
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
         break
    else:
        break


cv2.destroyAllWindows()
cap.release()
out.release()

