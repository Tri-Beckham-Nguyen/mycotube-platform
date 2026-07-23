# Time sync: ESP32 → Raspberry Pi 5

Keeping the stimulus controller and the logging host on the same clock, by
running the Pi as a local NTP server and having the ESP32 sync to it over WiFi.
This matters when timestamps from both devices have to be compared after a run.

> Network credentials below are placeholders. Fill in your own; do not commit
> real ones.

## Raspberry Pi 5 — run an NTP server with chrony

1. Install chrony:

       sudo apt update
       sudo apt install chrony

2. Edit the config:

       sudo nano /etc/chrony/chrony.conf

3. Allow LAN devices to query time (narrow the subnet to match your network):

       allow 192.168.0.0/16

4. Restart:

       sudo systemctl restart chrony

5. Confirm it's listening:

       chronyc sources

## ESP32 — sync to the Pi

```cpp
#include <WiFi.h>
#include <time.h>

const char* ssid      = "YOUR_WIFI_SSID";       // same network as the Pi
const char* password  = "YOUR_WIFI_PASSWORD";
const char* ntpServer = "YOUR_PI_IP_ADDRESS";   // the Pi's LAN address

const long gmtOffset_sec      = -18000;  // UTC-5 (US Eastern, standard time)
const int  daylightOffset_sec = 0;

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());

  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  Serial.print("Configured time with server: ");
  Serial.println(ntpServer);

  // Wait until the clock leaves epoch-1970 territory
  time_t now = time(nullptr);
  while (now < 24 * 3600) {
    delay(500);
    Serial.print(".");
    now = time(nullptr);
  }
  Serial.println("\nTime synchronized.");
}

void loop() {
  time_t now;
  time(&now);
  struct tm timeinfo;
  localtime_r(&now, &timeinfo);   // gmtime_r() for UTC instead

  Serial.print("Current local time: ");
  Serial.println(asctime(&timeinfo));
  delay(5000);
}
```

The `now < 24 * 3600` check is the useful part: an unsynced ESP32 reports a
timestamp near the Unix epoch, so waiting until the clock passes one day is a
simple way to block until NTP has actually landed.
