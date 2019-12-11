import cv2
import numpy as np
import matplotlib.pyplot as plt

cap=cv2.VideoCapture('circledetector.mp4')

while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(512,512)) 
    frame = cv2.transpose(frame)
    frame = cv2.flip(frame,1)
    print(frame.shape)
    cv2.imshow('initial',frame)
    pts1 = np.float32([[91,163],[392,163],[0,512],[512,512]])

    pts2= np.float32([[0,0],[300,0],[0,512],[300,512]])

    per_trans=cv2.getPerspectiveTransform(pts1,pts2)

    frame = cv2.warpPerspective(frame,per_trans,(300,512))

    blur = cv2.GaussianBlur(frame,(3,3),0)

    cv2.imshow('circle',blur)
    #hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    

    l_b = np.array([0,40,30])
    u_b = np.array([45,170,120])

    mask = cv2.inRange(frame, l_b, u_b)
    kernal=np.ones((1,1),np.uint8)

    mask=cv2.dilate(mask,kernal,iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    #cv2.imshow('mask', mask)
    #cv2.imshow('result',res)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=5, minRadius=35, maxRadius=80)
    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw inner circle
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)


    cv2.imshow('frame',frame)
    

    if cv2.waitKey(100) & 0xFF==ord('q'): #waitkey() waits for user input thing after amp symbol takes input
        break
cap.release()
cv2.destroyAllWindows()