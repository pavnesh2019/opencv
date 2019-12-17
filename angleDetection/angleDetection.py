import numpy as np 
import cv2

#cap=cv2.VideoCapture("opencv/hough/slow_coins.mp4")

cap = cv2.VideoCapture('coins.mp4')

while(True):
    ret, frame = cap.read()
    frame = cv2.medianBlur(frame,5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("check",frame)
    #cv2.waitKey(0)

###
#HughCircles Detection TEST  
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,50,
                          param1=50,param2=30,minRadius=110,maxRadius=150) 
    circles = np.uint16(np.around(circles))
    ret,thresh = cv2.threshold(gray,127,255,0)

 
# calculate moments of binary image
    M = cv2.moments(thresh)
 
# calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    if circles.all != None:
        for i in circles[0,:]:
        # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
            cv2.line(frame,(i[0],i[1]),(cX,cY),(0,0,0),2)
            thiselem = i
            
            theta1=np.arctan((thiselem[1]-cY)/(thiselem[0]-cX))
            theta1=theta1*180/np.pi
            
            print(theta1)
            frame=cv2.putText(frame,str(theta1),(thiselem[0],thiselem[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            if len(circles)>1:
                nextelem = circles[circles.index(i)-len(li)+1]
                theta2=np.arctan((cY-nextelem[1])/(cX-nextelem[0]))
                theta2=theta2*180/np.pi
                print(theta2) 
                frame=cv2.putText(frame,str(theta2),(nextelem[0],nextelem[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))

            
#    
###
 
# Display the resulting frame
    cv2.waitKey(500)
    cv2.imshow('live_hough_circlesq',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break