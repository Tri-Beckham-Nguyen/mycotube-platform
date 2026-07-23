# Recovering an ESP32 with a corrupted core dump partition

## Symptom

The board boot-loops, and the serial monitor shows something like:

    E (236) esp_core_dump_flash: Core dump flash config is corrupted!
    CRC=0x7bd5c66f instead of 0x0 Rebooting...
    ESP-ROM:esp32s3-20210327

## Fix

Erasing flash clears the corrupted partition and lets the board take a clean
upload.

1. Check the USB cable first — a charge-only cable will look like a dead board.
   Connect the ESP32 directly to the machine and note which COM port it enumerates on.

2. Erase the flash (adjust `--chip` and `--port` to match your board):

       python -m esptool --chip esp32s3 --port COM7 erase_flash

3. Re-flash the sketch normally.

## Prerequisites

If `esptool` isn't installed:

    pip install esptool

`esptool` needs Python on PATH. On Windows, install Python from python.org and
make sure the "Add Python to PATH" option is checked during install.
