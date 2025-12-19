import cv2

img = cv2.imread("test_ball.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

output = img.copy()

for contour in contours:
    area = cv2.contourArea(contour)
    if area < 200:  # skip tiny noise
        continue

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(approx)

    num_vertices = len(approx)

    if num_vertices == 3:
        shape_name = "Triangle"
    elif num_vertices == 4:
        shape_name = "Rect"
    else:
        shape_name = "Circle"

    # Draw contour and label
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(output, shape_name, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.imshow("Shapes Detected", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
