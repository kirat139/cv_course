import time
import cv2
import mediapipe as mp
import numpy as np


def overlay_canvas_on_frame(frame_bgr, canvas):
    """
    Overlay the drawing canvas on the live frame.
    Uses addWeighted: dst = frame*alpha + canvas*beta + gamma :contentReference[oaicite:3]{index=3}
    """
    return cv2.addWeighted(frame_bgr, 0.5, canvas, 1.0, 0.0)

def norm_conv(x, y, w, h):
    return int(x * w), int(y * h)

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Create a Hands object
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(1)
    finger_counter = 0
    finger_counter_2 = 0
    finger_counter_3 = 0
    finger_counter_4 = 0
    finger_counter_5 = 0
    prev_point = None
    overlay = None
    open_palm_start = None
    CLEAR_HOLD_SECONDS = 0.6 
    last_clear_time: float = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
                # Create canvas once we know frame size
        if overlay is None:
            overlay = np.zeros_like(frame)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame to get pose landmarks
        results = hands.process(rgb_frame)

        # Draw pose landmarks if present
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            h, w, c = frame.shape

            print("height of frame: ", h)
            print("width of frame: ", w)
            print("channel of frame: ", c)

            THUMB_IP = hand_landmarks.landmark[3]
            THUMB_TIP = hand_landmarks.landmark[4]
            INDEX_FINGER_PIP = hand_landmarks.landmark[6]
            INDEX_FINGER_TIP = hand_landmarks.landmark[8]
            MIDDLE_FINGER_PIP = hand_landmarks.landmark[10]
            MIDDLE_FINGER_TIP = hand_landmarks.landmark[12]
            RING_FINGER_PIP = hand_landmarks.landmark[14]
            RING_FINGER_TIP = hand_landmarks.landmark[16]
            PINKY_PIP = hand_landmarks.landmark[18]
            PINKY_TIP = hand_landmarks.landmark[20]

            ti_x, ti_y = int(THUMB_IP.x * w), int(THUMB_IP.y * h)
            tt_x, tt_y = int(THUMB_TIP.x * w), int(THUMB_TIP.y * h)
            ip_x, ip_y = int(INDEX_FINGER_PIP.x * w), int(INDEX_FINGER_PIP.y * h)
            it_x, it_y = int(INDEX_FINGER_TIP.x * w), int(INDEX_FINGER_TIP.y * h)
            mfp_x, mfp_y = int(MIDDLE_FINGER_PIP.x * w), int(MIDDLE_FINGER_PIP.y * h)
            mft_x, mft_y = int(MIDDLE_FINGER_TIP.x * w), int(MIDDLE_FINGER_TIP.y * h)
            rp_x, rp_y = int(RING_FINGER_PIP.x * w), int(RING_FINGER_PIP.y * h)
            rt_x, rt_y = int(RING_FINGER_TIP.x * w), int(RING_FINGER_TIP.y * h)
            pp_x, pp_y = int(PINKY_PIP.x * w), int(PINKY_PIP.y * h)
            pt_x, pt_y = int(PINKY_TIP.x * w), int(PINKY_TIP.y * h)

            # using function
            # ti_x, ti_y = norm_conv(THUMB_IP.x, THUMB_IP.y, w, h)
            # tt_x, tt_y = int(THUMB_TIP.x * w), int(THUMB_TIP.y * h)
            # ip_x, ip_y = int(INDEX_FINGER_PIP.x * w), int(INDEX_FINGER_PIP.y * h)
            # it_x, it_y = int(INDEX_FINGER_TIP.x * w), int(INDEX_FINGER_TIP.y * h)
            # mfp_x, mfp_y = int(MIDDLE_FINGER_PIP.x * w), int(MIDDLE_FINGER_PIP.y * h)
            # mft_x, mft_y = int(MIDDLE_FINGER_TIP.x * w), int(MIDDLE_FINGER_TIP.y * h)
            # rp_x, rp_y = int(RING_FINGER_PIP.x * w), int(RING_FINGER_PIP.y * h)
            # rt_x, rt_y = int(RING_FINGER_TIP.x * w), int(RING_FINGER_TIP.y * h)
            # pp_x, pp_y = int(PINKY_PIP.x * w), int(PINKY_PIP.y * h)
            # pt_x, pt_y = int(PINKY_TIP.x * w), int(PINKY_TIP.y * h)

            cv2.circle(frame, (it_x, it_y), 8, (255, 0, 0), -1)

            thumb_up = ti_x > tt_x
            index_finger_up = it_y > ip_y
            middle_finger_up = mft_y > mfp_y
            ring_finger_up = rt_y > rp_y
            pinky_finger_up = pt_y > pp_y

            if thumb_up:
                finger_counter_5=1
                text_5 = str(finger_counter_5)
                colour = (255,0,0)
            else:
                finger_counter_5 = 0
                text_5 = str(finger_counter_5)
                colour = (255,0,0)

            if index_finger_up:
                finger_counter=0
                text = str(finger_counter)
                colour = (255,0,0)
            else:
                finger_counter = 1
                text = str(finger_counter)
                colour = (255,0,0)

            if middle_finger_up:

                finger_counter_2=0
                text_2 = str(finger_counter_2)
                colour = (255,0,0)
            else:
                finger_counter_2 = 1
                text_2 = str(finger_counter_2)
                colour = (255,0,0)

            if ring_finger_up:
                finger_counter_3=0
                text_3 = str(finger_counter_3)
                colour = (255,0,0)
            else:
                finger_counter_3 = 1
                text_3 = str(finger_counter_3)
                colour = (255,0,0)   

            if pinky_finger_up:
                finger_counter_4=0
                text_4 = str(finger_counter_4)
                colour = (255,0,0)
            else:
                finger_counter_4 = 1
                text_4 = str(finger_counter_4)
                colour = (255,0,0)

            cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1.2, colour, 3)
            cv2.putText(frame, text_2, (50, 60), cv2.FONT_HERSHEY_COMPLEX, 1.2, colour, 3)
            cv2.putText(frame, text_3, (70, 60), cv2.FONT_HERSHEY_COMPLEX, 1.2, colour, 3)
            cv2.putText(frame, text_4, (90, 60), cv2.FONT_HERSHEY_COMPLEX, 1.2, colour, 3)
            cv2.putText(frame, text_5, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1.2, colour, 3)

            # Drawing Logic
            if finger_counter == 1 and finger_counter_2 == 0 and finger_counter_3 == 0 and finger_counter_4 == 0:
                cv2.putText(frame, "Pen: ON", (0, 180), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3)
                it_x, it_y = int(INDEX_FINGER_TIP.x * w), int(INDEX_FINGER_TIP.y * h)

                if prev_point is None:
                    prev_point = (it_x, it_y)
                else:
                    # Draw on canvas (not directly on frame)
                    cv2.line(overlay, prev_point, (it_x, it_y), (255,0,0),8)
                    prev_point = (it_x, it_y)  
            else:
                cv2.putText(frame, "Pen: OFF", (0, 180), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 3)
                prev_point = None     

            # Clearning Logic
            # ----- Clear gesture logic (hold open palm) -----
            now = time.time()    
            if finger_counter_5 == 0 and finger_counter == 1 and finger_counter_2 == 1 and finger_counter_3 == 1 and finger_counter_4 == 1:
                cv2.putText(frame, "clearing page", (0, 230), cv2.FONT_HERSHEY_COMPLEX, 1, (255,155,0), 3)         

                if open_palm_start is None:
                    open_palm_start = now
                else:
                    held = now - open_palm_start
                    cv2.putText(frame, f"OPEN PALM: hold {held:.1f}s", (10, 125),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    if held >= CLEAR_HOLD_SECONDS and (now - last_clear_time) > 1.0:
                        overlay[:] = 0
                        last_clear_time = now
                        open_palm_start = None
            else:
                open_palm_start = None

            # elif finger_counter_5 == 1:
            #         cv2.putText (frame, "close thumb to clear", (0, 230), cv2.FONT_HERSHEY_COMPLEX, 1, (255,155,0), 3)
            #         open_palm_start = None
        else:
            prev_point = None
            open_palm_start = None


        out = overlay_canvas_on_frame(frame,overlay)
        
        # cv2.imshow("MediaPipe finger", frame)
        cv2.imshow("numpy overlay", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
