import time
import cv2
import numpy as np
import mediapipe as mp


# ----------------------------
# A) Setup / Constants
# ----------------------------
CAMERA_INDEX = 0
MAX_HANDS = 1

DRAW_COLOR = (0, 255, 0)       # green in BGR
DRAW_THICKNESS = 8

CLEAR_HOLD_SECONDS = 0.6       # open palm hold duration
CLEAR_COOLDOWN_SECONDS = 1.0   # prevent repeated clears


# ----------------------------
# B) Camera Functions
# ----------------------------
def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open the webcam and return the camera object."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not found at index {camera_index}. Try 1.")
    return cap


def read_frame(cap: cv2.VideoCapture):
    """Read one frame from camera. Returns (success, frame)."""
    return cap.read()


def mirror_frame(frame):
    """Flip frame horizontally to behave like a selfie camera."""
    return cv2.flip(frame, 1)


# ----------------------------
# C) MediaPipe Functions
# ----------------------------
def create_hands_detector(max_hands: int):
    """
    Create and return MediaPipe Hands detector + mp_hands reference.
    MediaPipe Hands outputs:
      - multi_hand_landmarks: 21 landmarks per hand (x,y,z)
      - multi_handedness: Left/Right label
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    return hands, mp_hands


def bgr_to_rgb(frame_bgr):
    """Convert OpenCV BGR image to RGB (MediaPipe expects RGB)."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def detect_first_hand(hands, frame_bgr):
    """
    Detect hands and return:
      (hand_landmarks, hand_label, results)
    If no hand found, returns (None, None, results).
    """
    rgb = bgr_to_rgb(frame_bgr)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None, None, results

    hand_landmarks = results.multi_hand_landmarks[0]

    # Handedness label ("Left" or "Right")
    hand_label = None
    if results.multi_handedness and len(results.multi_handedness) > 0:
        hand_label = results.multi_handedness[0].classification[0].label
    else:
        hand_label = "Right"

    return hand_landmarks, hand_label, results


# ----------------------------
# D) Landmark / Drawing Functions
# ----------------------------
def ensure_canvas(canvas, frame_shape):
    """Create a black canvas (same size as frame) if canvas is None."""
    if canvas is None:
        return np.zeros(frame_shape, dtype=np.uint8)
    return canvas


def draw_hand_landmarks(frame_bgr, mp_hands, hand_landmarks):
    """Draw 21 landmarks and connections on the frame (for learning)."""
    mp_draw = mp.solutions.drawing_utils
    mp_draw.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)


def landmark_to_pixel(landmark, frame_w, frame_h):
    """
    MediaPipe x,y are normalized (0..1).
    Convert them to pixel coordinates.
    """
    x_px = int(landmark.x * frame_w)
    y_px = int(landmark.y * frame_h)
    return x_px, y_px


def draw_index_tip_dot(frame_bgr, x, y):
    """Draw a red dot at index fingertip to show the 'pen tip'."""
    cv2.circle(frame_bgr, (x, y), 10, (0, 0, 255), -1)


def draw_main_title(frame_bgr):
    """Draw title + instructions text on the frame."""
    cv2.putText(frame_bgr, "Air Drawing (Refactor for Class 9)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(frame_bgr, "Index UP + Middle DOWN = DRAW | Open Palm hold = CLEAR | C=Clear | Q=Quit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def show_pen_status(frame_bgr, pen_mode: bool):
    """Show PEN ON/OFF text."""
    if pen_mode:
        cv2.putText(frame_bgr, "PEN: ON", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame_bgr, "PEN: OFF", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def show_finger_debug(frame_bgr, count: int, hand_label: str):
    """Show finger count + handedness label (helps debug thumb logic)."""
    cv2.putText(frame_bgr, f"Fingers Up: {count} ({hand_label})", (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


# ----------------------------
# E) Gesture Functions
# ----------------------------
def finger_is_up(landmarks, tip_id: int, pip_id: int):
    """
    Finger up rule (index/middle/ring/pinky):
    If fingertip is higher than PIP joint => tip.y < pip.y
    (Smaller y means higher on screen.)
    """
    return landmarks[tip_id].y < landmarks[pip_id].y


def thumb_is_up(landmarks, hand_label: str):
    """
    Simple thumb rule using x direction + handedness.
    Note: MediaPipe handedness assumes mirrored image. :contentReference[oaicite:1]{index=1}
    - Right hand: thumb_tip.x > thumb_ip.x
    - Left hand : thumb_tip.x < thumb_ip.x
    """
    tip = landmarks[4]
    ip = landmarks[3]

    if hand_label.lower() == "right":
        return tip.x > ip.x
    return tip.x < ip.x


def get_fingers_up(landmarks, hand_label: str):
    """Return dict of finger states: True/False for each finger."""
    fingers = {}
    fingers["thumb"] = thumb_is_up(landmarks, hand_label)
    fingers["index"] = finger_is_up(landmarks, 8, 6)
    fingers["middle"] = finger_is_up(landmarks, 12, 10)
    fingers["ring"] = finger_is_up(landmarks, 16, 14)
    fingers["pinky"] = finger_is_up(landmarks, 20, 18)
    return fingers


def count_fingers_up(fingers_dict):
    """Count how many fingers are up (True)."""
    return sum(1 for v in fingers_dict.values() if v)


def is_pen_mode(fingers_dict):
    """Pen mode rule: Index UP and Middle DOWN."""
    return fingers_dict["index"] and (not fingers_dict["middle"])


def is_open_palm(fingers_dict):
    """Open palm means all 5 fingers up."""
    return count_fingers_up(fingers_dict) == 5


# ----------------------------
# F) Drawing on Canvas
# ----------------------------
def update_canvas_drawing(canvas, prev_point, curr_point, pen_mode,
                          color=DRAW_COLOR, thickness=DRAW_THICKNESS):
    """
    If pen_mode is True:
      - draw a line from prev_point to curr_point on the canvas
      - update prev_point
    If pen_mode is False:
      - reset prev_point so next stroke starts fresh
    Uses cv2.line to draw a segment between two points. :contentReference[oaicite:2]{index=2}
    """
    if not pen_mode:
        return canvas, None

    if prev_point is None:
        return canvas, curr_point

    cv2.line(canvas, prev_point, curr_point, color, thickness)
    return canvas, curr_point


# ----------------------------
# G) Clear Gesture (Open Palm Hold)
# ----------------------------
def update_clear_hold(canvas, open_palm, open_palm_start, last_clear_time,
                      hold_seconds=CLEAR_HOLD_SECONDS, cooldown=CLEAR_COOLDOWN_SECONDS):
    """
    Clear rule:
      - If open_palm becomes True, start timer.
      - If still open after hold_seconds, clear canvas.
      - Use cooldown to avoid repeated clears.
    Returns updated: (canvas, open_palm_start, last_clear_time, held_time)
    """
    now = time.time()

    if not open_palm:
        return canvas, None, last_clear_time, 0.0

    # palm is open
    if open_palm_start is None:
        return canvas, now, last_clear_time, 0.0

    held = now - open_palm_start
    can_clear = (held >= hold_seconds) and ((now - last_clear_time) > cooldown)

    if can_clear:
        canvas[:] = 0
        last_clear_time = now
        open_palm_start = None

    return canvas, open_palm_start, last_clear_time, held


def show_open_palm_hold(frame_bgr, held_seconds):
    """Show the open palm hold timer text."""
    if held_seconds > 0:
        cv2.putText(frame_bgr, f"OPEN PALM: hold {held_seconds:.1f}s", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


# ----------------------------
# H) Merge / Overlay Output
# ----------------------------
def overlay_canvas_on_frame(frame_bgr, canvas):
    """
    Overlay the drawing canvas on the live frame.
    Uses addWeighted: dst = frame*alpha + canvas*beta + gamma :contentReference[oaicite:3]{index=3}
    """
    return cv2.addWeighted(frame_bgr, 1.0, canvas, 1.0, 0.0)


# ----------------------------
# I) FPS Functions
# ----------------------------
def compute_fps(prev_time):
    """Compute FPS based on time difference and return (fps_int, new_prev_time)."""
    now = time.time()
    fps = 1.0 / max(1e-6, (now - prev_time))
    return int(fps), now


def draw_fps(frame_bgr, fps_int):
    """Draw FPS on the bottom-left."""
    h = frame_bgr.shape[0]
    cv2.putText(frame_bgr, f"FPS: {fps_int}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)


# ----------------------------
# J) Key Handling
# ----------------------------
def handle_keypress(key, canvas):
    """
    Handle keys:
      - q => quit
      - c => clear canvas
    Returns: (should_quit, canvas)
    """
    if key == ord('q'):
        return True, canvas
    if key == ord('c'):
        canvas[:] = 0
    return False, canvas


# ----------------------------
# K) MAIN (merge everything)
# ----------------------------
def main():
    cap = open_camera(CAMERA_INDEX)  # OpenCV webcam capture :contentReference[oaicite:4]{index=4}
    hands, mp_hands = create_hands_detector(MAX_HANDS)

    canvas = None
    prev_point = None
    open_palm_start = None
    last_clear_time = 0.0
    fps_prev_time = time.time()

    print("Running... Press Q to quit, C to clear.")

    while True:
        success, frame = read_frame(cap)
        if not success:
            break

        frame = mirror_frame(frame)

        # canvas same size as frame
        canvas = ensure_canvas(canvas, frame.shape)

        draw_main_title(frame)

        # Detect first hand
        hand_landmarks, hand_label, results = detect_first_hand(hands, frame)

        held_time = 0.0
        pen_mode = False

        if hand_landmarks is not None:
            draw_hand_landmarks(frame, mp_hands, hand_landmarks)

            h, w = frame.shape[:2]
            lm = hand_landmarks.landmark

            # Fingertip (index tip = 8)
            ix, iy = landmark_to_pixel(lm[8], w, h)
            draw_index_tip_dot(frame, ix, iy)

            # Gestures
            fingers = get_fingers_up(lm, hand_label)
            finger_count = count_fingers_up(fingers)

            pen_mode = is_pen_mode(fingers)
            show_pen_status(frame, pen_mode)
            show_finger_debug(frame, finger_count, hand_label)

            # Draw on canvas
            canvas, prev_point = update_canvas_drawing(
                canvas, prev_point, (ix, iy), pen_mode
            )

            # Clear gesture hold
            open_palm = is_open_palm(fingers)
            canvas, open_palm_start, last_clear_time, held_time = update_clear_hold(
                canvas, open_palm, open_palm_start, last_clear_time
            )
            show_open_palm_hold(frame, held_time)

        else:
            # No hand: stop stroke & reset palm timer
            prev_point = None
            open_palm_start = None

        # Merge canvas + frame
        out = overlay_canvas_on_frame(frame, canvas)

        # FPS
        fps, fps_prev_time = compute_fps(fps_prev_time)
        draw_fps(out, fps)

        cv2.imshow("Air Drawing - Function Version", out)

        key = cv2.waitKey(1) & 0xFF
        should_quit, canvas = handle_keypress(key, canvas)
        if should_quit:
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
