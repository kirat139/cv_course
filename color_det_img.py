# h = 0:179
# s = 0:255
# v = 0:255

import cv2
import numpy as np

img = cv2.imread("color_test.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Example blue range (rough)
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Original", img)
cv2.imshow("Mask", mask)      # white = detected areas
cv2.imshow("Result", result)  # only blue regions
cv2.waitKey(0)
cv2.destroyAllWindows()
