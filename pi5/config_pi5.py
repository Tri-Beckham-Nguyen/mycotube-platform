#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config_pi5.py — unified GPIO/SPI backend for ADS1263 on Raspberry Pi 5
Handles official libgpiod v2 (with naming-variant tolerance) and RPi.GPIO fallback.
"""

import time
import spidev

# BCM pin assignments (unchanged header layout on Pi 5)
RST_PIN = 18
CS_PIN  = 22
DRDY_PIN = 17

_spi = spidev.SpiDev()

# ----------------------------------------------------------------------
# Try gpiod first, then fall back to RPi.GPIO
# ----------------------------------------------------------------------
try:
    import gpiod
    _USE_GPIOD = True
except Exception:
    _USE_GPIOD = False


if _USE_GPIOD:
    try:
        # -------- libgpiod v2-ish API (tolerates naming differences) --------

        def _pick_attr(*paths, default=None):
            """
            Try attribute paths like:
              'LineDirection.OUTPUT'
              'line.Direction.OUTPUT'
              'Direction.OUTPUT'
            Returns the found value, else default.
            """
            for p in paths:
                obj = gpiod
                ok = True
                for part in p.split("."):
                    if not hasattr(obj, part):
                        ok = False
                        break
                    obj = getattr(obj, part)
                if ok:
                    return obj
            return default

        # Direction enums (vary across bindings)
        DIR_OUT = _pick_attr(
            "LineDirection.OUTPUT",
            "line.Direction.OUTPUT",
            "Direction.OUTPUT",
            default="output",
        )
        DIR_IN = _pick_attr(
            "LineDirection.INPUT",
            "line.Direction.INPUT",
            "Direction.INPUT",
            default="input",
        )

        # Bias enums (vary across bindings)
        BIAS_UP = _pick_attr(
            "LineBias.PULL_UP",
            "line.Bias.PULL_UP",
            "Bias.PULL_UP",
            default="pull-up",
        )

        # Value enums (needed for output_value!)
        VAL_ACTIVE = _pick_attr(
            "line.Value.ACTIVE",
            "Value.ACTIVE",
            default=1,
        )
        VAL_INACTIVE = _pick_attr(
            "line.Value.INACTIVE",
            "Value.INACTIVE",
            default=0,
        )

        _chip_path = "/dev/gpiochip0"

        # Some bindings like Chip instantiated (even if request_lines uses the path)
        _chip = gpiod.Chip(_chip_path)

        # Request lines with correct enum output values (ACTIVE == logical 1 unless active_low)
        _req = gpiod.request_lines(
            _chip_path,
            consumer="ads1263",
            config={
                CS_PIN: gpiod.LineSettings(direction=DIR_OUT, output_value=VAL_ACTIVE),
                RST_PIN: gpiod.LineSettings(direction=DIR_OUT, output_value=VAL_ACTIVE),
                DRDY_PIN: gpiod.LineSettings(direction=DIR_IN, bias=BIAS_UP),
            },
        )

        def digital_write(pin: int, value: int):
            _req.set_value(pin, VAL_ACTIVE if value else VAL_INACTIVE)

        def digital_read(pin: int) -> int:
            v = _req.get_value(pin)
            # v may be an enum or int; normalize to 0/1
            try:
                return int(v.value)
            except Exception:
                return int(v)

        def module_exit():
            try:
                _req.release()
            except Exception:
                pass

    except Exception as e:
        # -------- v1 fallback ONLY if get_line exists --------
        _chip_path = "/dev/gpiochip0"
        chip = gpiod.Chip(_chip_path)

        if not hasattr(chip, "get_line"):
            raise RuntimeError(
                "gpiod bindings lack expected v2 enums/APIs and also lack get_line() v1 API. "
                f"request_lines() failed with: {e}"
            )

        line_cs   = chip.get_line(CS_PIN)
        line_rst  = chip.get_line(RST_PIN)
        line_drdy = chip.get_line(DRDY_PIN)

        line_cs.request(consumer="ads1263",
                        type=gpiod.LINE_REQ_DIR_OUT,
                        default_vals=[1])
        line_rst.request(consumer="ads1263",
                         type=gpiod.LINE_REQ_DIR_OUT,
                         default_vals=[1])
        line_drdy.request(consumer="ads1263",
                          type=gpiod.LINE_REQ_DIR_IN)

        def digital_write(pin: int, value: int):
            if pin == CS_PIN:
                line_cs.set_value(1 if value else 0)
            elif pin == RST_PIN:
                line_rst.set_value(1 if value else 0)

        def digital_read(pin: int) -> int:
            return line_drdy.get_value() if pin == DRDY_PIN else 0

        def module_exit():
            line_cs.release()
            line_rst.release()
            line_drdy.release()

    # ---- Shared SPI helpers for gpiod paths ----
    def delay_ms(ms: int):
        time.sleep(ms / 1000.0)

    def spi_writebyte(data):
        _spi.xfer2(list(data))

    def spi_readbytes(n):
        return _spi.readbytes(n)

    def module_init() -> int:
        _spi.open(0, 0)               # CE0
        _spi.max_speed_hz = 2_000_000
        _spi.mode = 0b01              # SPI mode 1 (CPOL=0, CPHA=1)

        digital_write(CS_PIN, 1)      # CS idle high
        digital_write(RST_PIN, 1)     # RST idle high
        return 0


else:
    # -------- RPi.GPIO fallback --------
    import RPi.GPIO as GPIO

    def digital_write(pin, val):
        GPIO.output(pin, val)

    def digital_read(pin):
        return GPIO.input(pin)

    def delay_ms(ms):
        time.sleep(ms / 1000.0)

    def spi_writebyte(data):
        _spi.xfer2(list(data))

    def spi_readbytes(n):
        return _spi.readbytes(n)

    def module_init() -> int:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(RST_PIN, GPIO.OUT)
        GPIO.setup(CS_PIN, GPIO.OUT)
        GPIO.setup(DRDY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        _spi.open(0, 0)
        _spi.max_speed_hz = 2_000_000
        _spi.mode = 0b01

        GPIO.output(CS_PIN, 1)
        GPIO.output(RST_PIN, 1)
        return 0

    def module_exit():
        try:
            _spi.close()
        finally:
            GPIO.cleanup()
