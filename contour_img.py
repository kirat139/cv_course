import cv2

img = cv2.imread("test_ball.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Make binary image (white shapes, black background)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

print("Number of contours found:", len(contours))

# Draw contours in green
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

cv2.imshow("Original", img)
cv2.imshow("Threshold", thresh)
cv2.imshow("Contours", contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
