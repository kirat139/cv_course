import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def run_face_detection(frame_bgr, face_detector):
    """
    Takes a BGR frame, returns annotated frame.
    Uses MediaPipe Face Detection to detect face box + key points.
    """
    h, w, _ = frame_bgr.shape

    # MediaPipe expects RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    results = face_detector.process(frame_rgb)

    if results.detections:
        for det in results.detections:
            # Draws box + key points using MediaPipe helper
            mp_drawing.draw_detection(frame_bgr, det)

    return frame_bgr

