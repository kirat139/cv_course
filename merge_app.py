import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def map_range(value, in_min, in_max, out_min, out_max):
    """Map value from one range to another."""
    value = max(in_min, min(in_max, value))
    return out_min + (float(value - in_min) / float(in_max - in_min)) * (
        out_max - out_min
    )


def get_volume_controller_windows():
    """
    Returns (volume_interface, min_db, max_db) if Windows + pycaw available.
    Otherwise returns (None, None, None).
    """
    try:
        from pycaw.pycaw import AudioUtilities

        device = AudioUtilities.GetSpeakers()
        volume = device.EndpointVolume
        min_db, max_db, _ = volume.GetVolumeRange()
        return volume, min_db, max_db
    except Exception:
        return None, None, None


def run_face_detection(frame_bgr, face_detector):
    """
    Takes a BGR frame, returns annotated frame.
    Uses MediaPipe Face Detection to detect face box + key points.
    """
    h, w, _ = frame_bgr.shape

    # MediaPipe expects RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    results = face_detector.process(frame_rgb)
    print("results: ", results)

    if results.detections:
        for det in results.detections:
            print("det: ", det)
            # Draws box + key points using MediaPipe helper
            mp_drawing.draw_detection(frame_bgr, det)

    return frame_bgr


def run_face_mesh(frame_bgr, face_mesh):
    """
    Draws face mesh (tesselation + contours + irises) on the frame.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
            )
            mp_drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style(),
            )
    return frame_bgr


def run_gesture_volume(frame_bgr, hand_detector, volume_iface, min_db, max_db):
    """
    Detects hand, measures thumb-index distance, maps it to volume.
    Works as:
    - If pycaw available (Windows): sets real volume
    - Otherwise: only shows UI volume bar
    """
    hands, frame_bgr = hand_detector.findHands(frame_bgr, draw=True)

    vol_percent = None

    if hands:
        hand = hands[0]
        lm = hand["lmList"]  # 21 landmarks

        # Thumb tip = landmark 4, Index tip = landmark 8 (common convention)
        x1, y1 = lm[4][0], lm[4][1]
        x2, y2 = lm[8][0], lm[8][1]

        # Draw line
        cv2.line(frame_bgr, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.circle(frame_bgr, (x1, y1), 7, (255, 255, 255), cv2.FILLED)
        cv2.circle(frame_bgr, (x2, y2), 7, (255, 255, 255), cv2.FILLED)

        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        # These limits depend on camera distance;
        vol_percent = int(map_range(dist, 30, 200, 0, 100))

        # If Windows volume interface exists, map to dB and set volume
        if volume_iface is not None:
            vol_db = map_range(vol_percent, 0, 100, min_db, max_db)
            volume_iface.SetMasterVolumeLevel(
                vol_db, None
            )  # pycaw method :contentReference[oaicite:21]{index=21}

    # Draw volume bar UI (works on all OS)
    if vol_percent is None:
        vol_percent = 0

    bar_y = int(map_range(vol_percent, 0, 100, 400, 150))
    cv2.rectangle(frame_bgr, (50, 150), (85, 400), (255, 255, 255), 2)
    cv2.rectangle(frame_bgr, (50, bar_y), (85, 400), (255, 255, 255), cv2.FILLED)
    cv2.putText(
        frame_bgr,
        f"{vol_percent}%",
        (40, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return frame_bgr


def main():
    cap = cv2.VideoCapture(1)

    # MediaPipe objects
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    face_detector = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # CVZone hand detector
    hand_detector = HandDetector(detectionCon=0.7, maxHands=1)

    # Windows volume controller (optional)
    volume_iface, min_db, max_db = get_volume_controller_windows()

    mode = 1  # 1=Volume, 2=FaceDetect, 3=FaceMesh

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        if mode == 1:
            frame = run_gesture_volume(
                frame, hand_detector, volume_iface, min_db, max_db
            )
            cv2.putText(
                frame,
                "MODE 1: GESTURE VOLUME",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        elif mode == 2:
            frame = run_face_detection(frame, face_detector)
            cv2.putText(
                frame,
                "MODE 2: FACE DETECTION",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        elif mode == 3:
            frame = run_face_mesh(frame, face_mesh)
            cv2.putText(
                frame,
                "MODE 3: FACE MESH",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            frame,
            "Press 1/2/3 to change mode | q to quit",
            (10, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Class App", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            mode = 1
        elif key == ord("2"):
            mode = 2
        elif key == ord("3"):
            mode = 3

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
