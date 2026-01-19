import time
import serial

PORT = "COM3"
BAUD = 9600

ser = serial.Serial(PORT, BAUD, timeout=1)

# Some ESP32 boards reset/boot depending on RTS/DTR behavior.
# These lines often help prevent weird reset loops:
try:
    ser.setRTS(False)
    ser.setDTR(False)
except Exception:
    pass

# ESP32 often resets when serial opens -> wait so it can boot
time.sleep(2.5)

# Clean buffers
ser.reset_input_buffer()
ser.reset_output_buffer()

def send_cmd(cmd: str):
    # Always send newline because ESP32 reads until '\n'
    data = (cmd + "\n").encode("utf-8")
    print("PC -> ESP32:", repr(data))
    ser.write(data)
    ser.flush()

    # Read replies for a short window (STATUS/HELP can be multi-line)
    t0 = time.time()
    while time.time() - t0 < 0.7:
        line = ser.readline()
        if line:
            print("ESP32 -> PC:", line.decode(errors="ignore").strip())

print("Type commands: ON, OFF, BLINK 200, STOP, STATUS, HELP")
print("Type EXIT to quit.")

while True:
    cmd = input("Command> ").strip()
    if cmd.upper() == "EXIT":
        break
    if cmd == "":
        continue
    send_cmd(cmd)

ser.close()
print("Disconnected.")
