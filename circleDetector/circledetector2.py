import cv2
import numpy as np




img = cv2.imread('circledetector4.jpg', cv2.IMREAD_COLOR)
# Convert to gray-scale
img=cv2.resize(img,(225,400))
#print(img.shape[0]/64)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image to reduce noise
img_blur = cv2.medianBlur(gray, 5)
# Apply hough transform on the image
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=5, minRadius=10, maxRadius=50)
# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)


cv2.imshow('circles',img)
cv2.waitKey(0)



cv2.destroyAllWindows()
cap.release()
out.release()
