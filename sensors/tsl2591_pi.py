
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal TSL2591 driver for Raspberry Pi (I2C, smbus2).
- Default address: 0x29
- Provides init(), read_channels(), and read_lux() (simple approximation)
- Allows setting integration time and gain.
"""

import time
from smbus2 import SMBus

TSL2591_ADDR = 0x29

# Command bit per datasheet
CMD = 0xA0

# Registers
REG_ENABLE  = 0x00
REG_CONTROL = 0x01
REG_C0DATAL = 0x14  # CH0 low, then +1 high
REG_C1DATAL = 0x16  # CH1 low, then +1 high

# ENABLE bits
EN_PON = 0x01   # Power ON
EN_AEN = 0x02   # ALS enable

# Integration time options (ALS)
# Value is written directly to CONTROL[2:0]; see datasheet mapping.
ITIME_CODES = {
    '100ms': 0x00,
    '200ms': 0x01,
    '300ms': 0x02,
    '400ms': 0x03,
    '500ms': 0x04,
    '600ms': 0x05,
}
# Gain options CONTROL[5:4]
GAIN_CODES = {
    'LOW':   0x00,  # 1x
    'MED':   0x10,  # 25x (per datasheet)
    'HIGH':  0x20,  # 428x
    'MAX':   0x30,  # 9876x
}

class TSL2591:
    def __init__(self, bus: int = 1, addr: int = TSL2591_ADDR,
                 itime: str = '100ms', gain: str = 'LOW'):
        self.addr = addr
        self.bus = SMBus(bus)
        self.itime = itime if itime in ITIME_CODES else '100ms'
        self.gain = gain if gain in GAIN_CODES else 'LOW'
        self._enabled = False

    def _write8(self, reg, val):
        self.bus.write_byte_data(self.addr, CMD | reg, val & 0xFF)

    def _read16(self, reg_low):
        lo = self.bus.read_byte_data(self.addr, CMD | reg_low)
        hi = self.bus.read_byte_data(self.addr, CMD | (reg_low + 1))
        return (hi << 8) | lo

    def enable(self):
        # Power on sequence
        self._write8(REG_ENABLE, EN_PON)
        time.sleep(0.003)
        self._write8(REG_ENABLE, EN_PON | EN_AEN)
        self._enabled = True

    def disable(self):
        self._write8(REG_ENABLE, 0x00)
        self._enabled = False

    def set_config(self, itime: str = None, gain: str = None):
        if itime: self.itime = itime if itime in ITIME_CODES else self.itime
        if gain:  self.gain  = gain if gain in GAIN_CODES else self.gain
        ctrl = (GAIN_CODES[self.gain] & 0x30) | (ITIME_CODES[self.itime] & 0x07)
        self._write8(REG_CONTROL, ctrl)

    def init(self):
        self.enable()
        self.set_config(self.itime, self.gain)

    def read_channels(self):
        # Return (full_spectrum, ir) raw 16-bit counts
        c0 = self._read16(REG_C0DATAL)
        c1 = self._read16(REG_C1DATAL)
        return c0, c1

    def read_lux(self):
        """
        Simple lux approximation based on raw channels.
        For rigorous lux computation, sensor-specific coefficients should be used.
        Here we provide a reasonable heuristic: visible = c0 - c1, scale for itime/gain.
        """
        c0, c1 = self.read_channels()
        visible = max(0, c0 - c1)

        # Scale by integration time and gain to approximate lux
        it_ms = int(self.itime.replace('ms',''))
        gain_map = {'LOW':1.0,'MED':25.0,'HIGH':428.0,'MAX':9876.0}
        gain = gain_map[self.gain]

        # counts per ms per gain (heuristic normalization constant)
        # Adjust the constant if you want closer absolute lux.
        norm = (visible / max(1.0, (it_ms * gain)))
        lux = norm * 5.0  # heuristic scale factor
        return lux, (c0, c1)

    def close(self):
        try:
            self.disable()
        finally:
            self.bus.close()
