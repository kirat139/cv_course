import cv2
import numpy as np
import time


def capture_background(cap, num_frames: int = 60, delay_sec: float = 2.0):
    """
    Capture a clean background frame.
    Assumes the subject is NOT in front of the camera during capture.
    """
    print("[INFO] Warming up camera...")
    time.sleep(delay_sec)

    background = None
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        # Flip frame for natural (mirror-like) viewing
        frame = cv2.flip(frame, 1)
        background = frame
    print("[INFO] Background captured.")
    return background


def main():
    # Open default webcam
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    # Step 1: Capture the background (ask user to move out of frame)
    print("Please move out of the frame. Capturing background...")
    background = capture_background(cap)

    print("[INFO] You can now step into the frame with the GREEN cloak.")
    print("[INFO] Press 'q' to quit.")

    # Pre-create a kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ------------------------------------------------------------------
        # Define HSV range for GREEN cloak
        # NOTE: You may need to tweak these bounds depending on lighting and
        #       the exact shade of green in the cloth.
        #
        # Typical example range for green in HSV:
        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])
        # ------------------------------------------------------------------

        # Create mask to detect green color
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological operations to remove noise and smooth the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        # Inverse mask: areas that are NOT green
        mask_inv = cv2.bitwise_not(mask)

        # Keep only the parts of the current frame that are NOT green
        res1 = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Take only the green parts from the background (to "fill in" cloak area)
        res2 = cv2.bitwise_and(background, background, mask=mask)

        # Combine res1 and res2 to get final output
        final = cv2.addWeighted(res1, 1, res2, 1, 0)

        cv2.putText(
            final,
            "Harry Potter Invisible Cloak (Green)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Magic Cloak", final)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
