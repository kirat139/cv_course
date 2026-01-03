import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(1)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame to get pose landmarks
    results = pose.process(rgb_frame)

    # Draw pose landmarks if present
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        h, w, c = frame.shape

        print("height of frame: ", h)
        print("width of frame: ", w)
        print("channel of frame: ", c)

        # l11 = left_shoulder, l12 = right_shoulder
        # l15 = left_wrist, l16 = right_wrist

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        ls_x, ls_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
        rs_x, rs_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
        lw_x, lw_y = int(left_wrist.x * w), int(left_wrist.y * h)
        rw_x, rw_y = int(right_wrist.x * w), int(right_wrist.y * h)


        print("left_shoulder: ", left_shoulder, ls_x, ls_y)
        print("right_shoulder: ", right_shoulder, rs_x, rs_y )
        print("left_wrist: ", left_wrist, lw_x, lw_y)
        print("right_wrist: ", right_wrist, rw_x, rw_y )

        cv2.circle(frame, (ls_x, ls_y), 8, (255, 0, 0), -1)
        cv2.circle(frame, (rs_x, rs_y), 8, (255, 0, 0), -1)
        cv2.circle(frame, (lw_x, lw_y), 8, (0, 255, 0), -1)
        cv2.circle(frame, (rw_x, rw_y), 8, (0, 255, 0), -1)

        hands_up = lw_y < ls_y and rw_y < rs_y

        if hands_up:
            text = "Hands UP!!"
            color = (0, 255, 0)
        else:
            text = "Hands DOWN!!"
            color = (0, 0, 255)

        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1.2, color, 3)

        # mp_drawing.draw_landmarks(
        #     frame,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS
        # )

    cv2.imshow("MediaPipe Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()
