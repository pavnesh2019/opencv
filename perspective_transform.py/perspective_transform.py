import cv2
import numpy as np

def nothing(x):
    pass

#cap=cv2.VideoCapture(0)






#while True:
frame=cv2.imread('perspective.jpeg')
#_, frame = cap.read()
cv2.imshow('image',frame)
pts1 = np.float32([[430,320],[590,320],[120,450],[850,450]])

pts2= np.float32([[0,0],[999,0],[0,666],[999,666]])

per_trans=cv2.getPerspectiveTransform(pts1,pts2)

perspective = cv2.warpPerspective(frame,per_trans,(999,666))


frame=cv2.resize(frame,(512,512))
cv2.imshow('image',frame)


perGray=cv2.cvtColor(perspective,cv2.COLOR_BGR2GRAY)
kernel = np.ones((1,1),np.float32)/25
frame = cv2.filter2D(perGray,-1,kernel)
result=cv2.resize(perGray,(512,512))
cv2.imshow('image',frame)
kernel=np.ones((5,5),np.uint8)

mask=cv2.erode(result,kernel)
cv2.imshow("mask",mask)

lt=30
ut=200

edges=cv2.Canny(mask,lt,ut)
cv2.imshow('gray',edges)
cv2.imshow('perspective transformed',perGray)


key=cv2.waitKey(0)

#cap.release()
cv2.destroyAllWindows()