import time
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


# -----------------------------
# Class 8: Air Drawing Settings
# -----------------------------
CAMERA_INDEX = 1            # change to 1 if webcam doesn't open
MAX_HANDS = 1               # keep 1 for school kid simplicity
DRAW_THICKNESS = 8
DRAW_COLOR = (0, 255, 0)    # green (BGR)
TEXT_COLOR = (255, 255, 255)

# Gesture tuning
CLEAR_HOLD_SECONDS = 0.6    # open palm must be held this long to clear (prevents accidental clears)


# -----------------------------
# Helper functions
# -----------------------------
def norm_to_pixel(lm, frame_w: int, frame_h: int) -> Tuple[int, int]:
    """Convert normalized landmark (0..1) to pixel coords."""
    return int(lm.x * frame_w), int(lm.y * frame_h)


def finger_is_up(landmarks, tip_id: int, pip_id: int) -> bool:
    """
    For index/middle/ring/pinky:
    If tip is above pip (smaller y), finger is 'up'.
    """
    return landmarks[tip_id].y < landmarks[pip_id].y


def thumb_is_up(landmarks, hand_label: str) -> bool:
    """
    Simple thumb heuristic using x direction + handedness label.
    - For Right hand: thumb tip x > thumb IP x  => thumb open
    - For Left hand : thumb tip x < thumb IP x  => thumb open

    Note: This is a practical heuristic; thumb can be tricky for different rotations.
    """
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    if hand_label.lower() == "right":
        return thumb_tip.x > thumb_ip.x
    else:
        return thumb_tip.x < thumb_ip.x


def get_hand_label(results, hand_index: int) -> str:
    """
    Read handedness label from MediaPipe.
    Usually results.multi_handedness[hand_index].classification[0].label
    """
    if results.multi_handedness and len(results.multi_handedness) > hand_index:
        return results.multi_handedness[hand_index].classification[0].label
    return "Right"  # fallback


# -----------------------------
# Main program
# -----------------------------
def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}. Try CAMERA_INDEX=1.")
        return

    prev_point: Optional[Tuple[int, int]] = None
    canvas: Optional[np.ndarray] = None

    open_palm_start: Optional[float] = None
    last_clear_time: float = 0.0

    fps_prev_time = time.time()

    print("Class 8 Air Drawing started.")
    print("Controls: Q = Quit | C = Clear")
    print("Gesture: Index UP + Middle DOWN => Draw | Open Palm (5 up) hold => Clear")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror view for the student
        h, w = frame.shape[:2]

        # Create canvas once we know frame size
        if canvas is None:
            canvas = np.zeros_like(frame)

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # UI text
        cv2.putText(frame, "Air Drawing (Class 8)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2)
        cv2.putText(frame, "Index UP + Middle DOWN = DRAW | Open Palm hold = CLEAR | C=Clear | Q=Quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2)

        pen_mode = False
        open_palm = False

        if results.multi_hand_landmarks:
            # Use first detected hand
            hand_lms = results.multi_hand_landmarks[0]
            landmarks = hand_lms.landmark

            hand_label = get_hand_label(results, 0)

            # Draw the landmarks on the live frame (for learning)
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Finger states
            index_up = finger_is_up(landmarks, tip_id=8, pip_id=6)
            middle_up = finger_is_up(landmarks, tip_id=12, pip_id=10)
            ring_up = finger_is_up(landmarks, tip_id=16, pip_id=14)
            pinky_up = finger_is_up(landmarks, tip_id=20, pip_id=18)
            thumb_up = thumb_is_up(landmarks, hand_label)

            fingers_up_count = sum([thumb_up, index_up, middle_up, ring_up, pinky_up])

            # Pen mode rule (more stable than "index up only"):
            # Index UP + Middle DOWN
            pen_mode = index_up and (not middle_up)

            # Open palm rule:
            open_palm = (fingers_up_count == 5)

            # Current index fingertip pixel position
            ix, iy = norm_to_pixel(landmarks[8], w, h)

            # Show a circle at the index tip
            cv2.circle(frame, (ix, iy), 10, (0, 0, 255), -1)

            # ----- Drawing logic -----
            if pen_mode:
                cv2.putText(frame, "PEN: ON", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if prev_point is None:
                    prev_point = (ix, iy)
                else:
                    # Draw on canvas (not directly on frame)
                    cv2.line(canvas, prev_point, (ix, iy), DRAW_COLOR, DRAW_THICKNESS)
                    prev_point = (ix, iy)
            else:
                cv2.putText(frame, "PEN: OFF", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                prev_point = None

            # ----- Clear gesture logic (hold open palm) -----
            now = time.time()
            if open_palm:
                if open_palm_start is None:
                    open_palm_start = now
                else:
                    held = now - open_palm_start
                    cv2.putText(frame, f"OPEN PALM: hold {held:.1f}s", (10, 125),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    if held >= CLEAR_HOLD_SECONDS and (now - last_clear_time) > 1.0:
                        canvas[:] = 0
                        last_clear_time = now
                        open_palm_start = None
            else:
                open_palm_start = None

            # Extra debug (optional): show finger count
            cv2.putText(frame, f"Fingers Up: {fingers_up_count} ({hand_label})", (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        else:
            prev_point = None
            open_palm_start = None

        # Blend canvas on top of frame
        # (canvas keeps the drawing persistent)
        out = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0.0)

        # FPS display (nice for teaching)
        fps_now_time = time.time()
        fps = 1.0 / max(1e-6, (fps_now_time - fps_prev_time))
        fps_prev_time = fps_now_time
        cv2.putText(out, f"FPS: {int(fps)}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        cv2.imshow("Class 8 - Air Drawing", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c') and canvas is not None:
            canvas[:] = 0

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
