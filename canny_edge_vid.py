import cv2

cap = cv2.VideoCapture(1)

while True:
    success, frame = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    cv2.imshow("Webcam - Original", frame)
    cv2.imshow("Webcam - Gray", gray)
    cv2.imshow("Webcam - Blur", blur)
    cv2.imshow("Webcam - Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
