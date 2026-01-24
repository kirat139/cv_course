import cv2
import time
import serial
import numpy as np
from cvzone.HandTrackingModule import HandDetector


PORT = "COM3"      # change if needed
BAUD = 9600
COOLDOWN_SEC = 0.02  # debounce (prevents spamming commands)

# -----------------------------
# SERIAL SETUP (pySerial)
# timeout prevents blocking forever on reads :contentReference[oaicite:10]{index=10}
# -----------------------------
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2.0)  # ESP32 often resets when port opens; wait a bit

ser.reset_input_buffer()
ser.reset_output_buffer()

# -----------------------------
# CV SETUP
# -----------------------------
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.7, maxHands=1)

last_cmd = None
last_sent_time = 0
last_reply = ""

def gesture_to_command(fingers):
    """
    fingers is like [thumb,index,middle,ring,pinky]
    Return (gesture_name, list_of_commands)
    """
    up_count = sum(fingers)

    # 0 fingers -> OFF
    if up_count == 0:
        return "FIST (0) => OFF", ["OFF"]

    # 1 finger (index only) -> ON
    if fingers == [0, 1, 0, 0, 0]:
        return "INDEX (1) => ON", ["ON"]

    # 2 fingers (index+middle) -> BLINK 200
    if fingers == [0, 1, 1, 0, 0]:
        return "TWO (2) => BLINK 200", ["BLINK 200"]

    # 5 fingers -> STOP then STATUS
    if up_count == 5:
        return "PALM (5) => STOP + STATUS", ["STOP", "STATUS"]

    return f"OTHER ({fingers}) => NO ACTION", []

def send_command(cmd):
    """
    Send a single command and attempt to read replies quickly.
    pySerial write expects bytes, so we encode; newline ends the command line. :contentReference[oaicite:11]{index=11}
    """
    global last_reply
    data = (cmd + "\n").encode("utf-8")
    ser.write(data)
    ser.flush()

    # Read a few lines (STATUS/HELP can be multi-line)
    replies = []
    start = time.time()
    while time.time() - start < 0.02:
        line = ser.readline()  # readline needs timeout to avoid hanging :contentReference[oaicite:12]{index=12}
        if line:
            replies.append(line.decode(errors="ignore").strip())

    if replies:
        last_reply = " | ".join(replies[-3:])  # keep last few lines

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)

    hands, frame = detector.findHands(frame, draw=True)
    gesture_name = "NO HAND"
    commands = []

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        gesture_name, commands = gesture_to_command(fingers)

        # Debounce: only send if command changed and cooldown passed
        now = time.time()
        if commands:
            cmd_key = ",".join(commands)
            if (cmd_key != last_cmd) and (now - last_sent_time >= COOLDOWN_SEC):
                for c in commands:
                    send_command(c)
                last_cmd = cmd_key
                last_sent_time = now

    # UI overlay
    cv2.putText(frame, f"Gesture: {gesture_name}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Last Sent: {last_cmd}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"ESP32 Reply: {last_reply}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press q to quit", (20, 680),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Class 13: Gesture to ESP32 Control", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
