#include <WiFi.h>
#include <WebServer.h>

// ---------- Wi-Fi AP credentials ----------
const char* AP_SSID = "ESP32-REMOTE";
const char* AP_PASS = "12345678";   // min 8 chars for WPA2 in many setups

#ifndef LED_BUILTIN
#define LED_BUILTIN 2
#endif

WebServer server(80);

// ---------- LED / Blink state ----------
bool ledOn = false;
bool blinkMode = false;
int blinkMs = 500;

unsigned long lastToggle = 0;

// ---------- Helper: update real LED pin ----------
void applyLed(bool on) {
  ledOn = on;
  digitalWrite(LED_BUILTIN, on ? HIGH : LOW);
}

// ---------- HTML page ----------
String makePage() {
  String s = "";
  s += "<!doctype html><html><head><meta name='viewport' content='width=device-width, initial-scale=1'>";
  s += "<title>ESP32 Remote</title></head><body style='font-family:Arial; text-align:center; padding:20px;'>";
  s += "<h2>ESP32 Wireless Remote</h2>";

  s += "<p><b>Status:</b> LED=" + String(ledOn ? "ON" : "OFF");
  s += " | Blink=" + String(blinkMode ? "ON" : "OFF");
  s += " | ms=" + String(blinkMs) + "</p>";

  s += "<p><a href='/on'><button style='font-size:22px;padding:12px 24px;'>ON</button></a></p>";
  s += "<p><a href='/off'><button style='font-size:22px;padding:12px 24px;'>OFF</button></a></p>";
  s += "<p><a href='/blink?ms=200'><button style='font-size:22px;padding:12px 24px;'>BLINK 200</button></a></p>";
  s += "<p><a href='/blink?ms=700'><button style='font-size:22px;padding:12px 24px;'>BLINK 700</button></a></p>";
  s += "<p><a href='/stop'><button style='font-size:22px;padding:12px 24px;'>STOP</button></a></p>";
  s += "<p><a href='/status'><button style='font-size:22px;padding:12px 24px;'>STATUS (text)</button></a></p>";

  s += "<hr><p style='font-size:14px;'>Try URL: /on /off /blink?ms=200 /status</p>";
  s += "</body></html>";
  return s;
}

// ---------- Routes ----------
void handleRoot() {
  server.send(200, "text/html", makePage());
}

void handleOn() {
  blinkMode = false;
  applyLed(true);
  server.send(200, "text/html", makePage());
}

void handleOff() {
  blinkMode = false;
  applyLed(false);
  server.send(200, "text/html", makePage());
}

void handleStop() {
  blinkMode = false;
  applyLed(false);
  server.send(200, "text/html", makePage());
}

void handleBlink() {
  // Read ms from query: /blink?ms=200
  // GET URLs commonly carry key=value pairs in the query string. :contentReference[oaicite:10]{index=10}
  if (server.hasArg("ms")) {
    int ms = server.arg("ms").toInt();
    if (ms >= 50 && ms <= 5000) {
      blinkMs = ms;
      blinkMode = true;
    }
  }
  server.send(200, "text/html", makePage());
}

void handleStatus() {
  String t = "";
  t += "LED=" + String(ledOn ? "ON" : "OFF");
  t += " Blink=" + String(blinkMode ? "ON" : "OFF");
  t += " ms=" + String(blinkMs);
  server.send(200, "text/plain", t);
}

void handleNotFound() {
  server.send(404, "text/plain", "Not found. Try /, /on, /off, /blink?ms=200, /status");
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  applyLed(false);

  Serial.begin(9600);

  // Start Wi-Fi Access Point (hotspot)
  // The Arduino-ESP32 docs include an AP example and web server usage flow. :contentReference[oaicite:11]{index=11}
  WiFi.softAP(AP_SSID, AP_PASS);

  IPAddress ip = WiFi.softAPIP();
  Serial.print("AP IP: ");
  Serial.println(ip); // typically 192.168.4.1

  // Routes
  server.on("/", handleRoot);
  server.on("/on", handleOn);
  server.on("/off", handleOff);
  server.on("/stop", handleStop);
  server.on("/blink", handleBlink);
  server.on("/status", handleStatus);
  server.onNotFound(handleNotFound);

  server.begin();
  Serial.println("HTTP server started. Open http://192.168.4.1/ in phone browser.");
}

void loop() {
  // Handle incoming HTTP requests
  server.handleClient();

  // Non-blocking blink (so server stays responsive)
  if (blinkMode) {
    unsigned long now = millis();
    if (now - lastToggle >= (unsigned long)blinkMs) {
      lastToggle = now;
      ledOn = !ledOn;
      digitalWrite(LED_BUILTIN, ledOn ? HIGH : LOW);
    }
  }
}
