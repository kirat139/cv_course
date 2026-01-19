#ifndef LED_BUILTIN
#define LED_BUILTIN 2   // common ESP32 dev board LED pin
#endif

bool blinkMode = false;
int blinkIntervalMs = 500;
unsigned long lastToggleMs = 0;
bool ledState = false;

void setLed(bool on) {
  ledState = on;
  digitalWrite(LED_BUILTIN, on ? HIGH : LOW);
}

void printHelp() {
  Serial.println("Commands:");
  Serial.println("  ON");
  Serial.println("  OFF");
  Serial.println("  BLINK <ms>  (50..5000)");
  Serial.println("  STOP");
  Serial.println("  STATUS");
  Serial.println("  HELP");
}

void printStatus() {
  Serial.print("blinkMode=");
  Serial.println(blinkMode ? "ON" : "OFF");

  Serial.print("blinkIntervalMs=");
  Serial.println(blinkIntervalMs);

  Serial.print("ledState=");
  Serial.println(ledState ? "HIGH" : "LOW");
}

void handleCommand(String cmd) {
  cmd.trim();          // removes spaces + removes \r if present
  cmd.toUpperCase();

  Serial.print("CMD RX: ");   // debug: what exactly arrived
  Serial.println(cmd);

  if (cmd == "ON") {
    blinkMode = false;
    setLed(true);
    Serial.println("OK: LED ON");
  }
  else if (cmd == "OFF") {
    blinkMode = false;
    setLed(false);
    Serial.println("OK: LED OFF");
  }
  else if (cmd.startsWith("BLINK")) {
    int spaceIndex = cmd.indexOf(' ');
    if (spaceIndex == -1) {
      Serial.println("ERROR: Use BLINK <ms> (example: BLINK 200)");
      return;
    }
    int ms = cmd.substring(spaceIndex + 1).toInt();
    if (ms < 50 || ms > 5000) {
      Serial.println("ERROR: blink ms must be 50..5000");
      return;
    }
    blinkIntervalMs = ms;
    blinkMode = true;
    Serial.print("OK: BLINK interval=");
    Serial.println(blinkIntervalMs);
  }
  else if (cmd == "STOP") {
    blinkMode = false;
    setLed(false);
    Serial.println("OK: STOP");
  }
  else if (cmd == "STATUS") {
    printStatus();
  }
  else if (cmd == "HELP" || cmd == "?") {
    printHelp();
  }
  else {
    Serial.print("ERROR: Unknown command: ");
    Serial.println(cmd);
    Serial.println("Type HELP");
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);

  Serial.begin(9600);
  Serial.setTimeout(300);     // affects readStringUntil timing

  delay(1000);                // helps if Serial Monitor connects slightly late
  Serial.println("ESP32 READY. Send commands ending with \\n");
  printHelp();
}

void loop() {
  // Read a full line command (Python must send newline \n)
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }

  // Non-blocking blink
  if (blinkMode) {
    unsigned long now = millis();
    if (now - lastToggleMs >= (unsigned long)blinkIntervalMs) {
      lastToggleMs = now;
      ledState = !ledState;
      digitalWrite(LED_BUILTIN, ledState ? HIGH : LOW);
    }
  }
}
