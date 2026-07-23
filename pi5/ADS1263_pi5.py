#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADS1263_pi5.py — Raspberry Pi 5 driver for TI ADS1263 (32-bit ADC).

- Uses config_pi5 backend (gpiod v2 on Pi 5, RPi.GPIO fallback).
- Explicit, datasheet-aligned register and command definitions.
- Full control of:
    * Digital filter type (Sinc1/2/3/4, FIR)
    * Data rate
    * PGA gain and bypass
    * Chop mode (off, chop, IDAC rotation, both)
    * Conversion delay
    * INTERFACE register: STATUS byte, checksum, timeout
- Correct ADC1/ADC2 read-by-command handling (frame length depends on INTERFACE).
- Multi-differential “scan” support for ADC1 with first-sample discard.
- INPMUX write-verification to catch “stuck MUX” / mirrored-channel issues.

Notes:
- ADS1263 can only convert **one differential pair at a time**. There is no
  hardware multi-diff scan mode; we implement a software scan by:
    1) Programming INPMUX for a given pair.
    2) Discarding the first conversion after the change.
    3) Using the second conversion as the settled value.
"""

from __future__ import annotations

import time
from typing import Iterable, List, Sequence

import config_pi5 as config  # Must provide: RST_PIN, CS_PIN, DRDY_PIN, digital_write/read, spi_writebyte/readbytes, delay_ms, module_init, module_exit

# ---------------------------------------------------------------------------
# Register map (ADC1 + shared + ADC2) — Table 38 of TI datasheet.
# ---------------------------------------------------------------------------
ADS1263_REG = {
    'REG_ID'       : 0x00,
    'REG_POWER'    : 0x01,
    'REG_INTERFACE': 0x02,
    'REG_MODE0'    : 0x03,
    'REG_MODE1'    : 0x04,
    'REG_MODE2'    : 0x05,
    'REG_INPMUX'   : 0x06,
    'REG_OFCAL0'   : 0x07,
    'REG_OFCAL1'   : 0x08,
    'REG_OFCAL2'   : 0x09,
    'REG_FSCAL0'   : 0x0A,
    'REG_FSCAL1'   : 0x0B,
    'REG_FSCAL2'   : 0x0C,
    'REG_IDACMUX'  : 0x0D,
    'REG_IDACMAG'  : 0x0E,
    'REG_REFMUX'   : 0x0F,
    'REG_TDACP'    : 0x10,
    'REG_TDACN'    : 0x11,
    'REG_GPIOCON'  : 0x12,
    'REG_GPIODIR'  : 0x13,
    'REG_GPIODAT'  : 0x14,

    # ADC2 specific
    'REG_ADC2CFG'  : 0x15,
    'REG_ADC2MUX'  : 0x16,
    'REG_ADC2OFC0' : 0x17,
    'REG_ADC2OFC1' : 0x18,
    'REG_ADC2FSC0' : 0x19,
    'REG_ADC2FSC1' : 0x1A,
}

# ---------------------------------------------------------------------------
# Command opcodes — Table 37 of TI datasheet.
# ---------------------------------------------------------------------------
ADS1263_CMD = {
    'CMD_NOP'     : 0x00,
    'CMD_RESET'   : 0x06,  # 0000 011x (06h or 07h)
    'CMD_START1'  : 0x08,
    'CMD_STOP1'   : 0x0A,
    'CMD_START2'  : 0x0C,
    'CMD_STOP2'   : 0x0E,
    'CMD_RDATA1'  : 0x12,
    'CMD_RDATA2'  : 0x14,
    'CMD_SYOCAL1' : 0x16,
    'CMD_SYGCAL1' : 0x17,
    'CMD_SFOCAL1' : 0x19,
    'CMD_SYOCAL2' : 0x1B,
    'CMD_SYGCAL2' : 0x1C,
    'CMD_SFOCAL2' : 0x1E,
    # RREG/WREG: base opcodes; lower 5 bits = register address
    'CMD_RREG'    : 0x20,
    'CMD_WREG'    : 0x40,
}

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# PGA gain codes (MODE2[6:4])
ADS1263_GAIN = {
    'ADS1263_GAIN_1' : 0,
    'ADS1263_GAIN_2' : 1,
    'ADS1263_GAIN_4' : 2,
    'ADS1263_GAIN_8' : 3,
    'ADS1263_GAIN_16': 4,
    'ADS1263_GAIN_32': 5,
    # 6,7 reserved
}

# ADC1 data rate codes (MODE2[3:0]) — see datasheet, Mode2 register.
ADS1263_DRATE = {
    'ADS1263_2d5SPS' : 0x0,
    'ADS1263_5SPS'   : 0x1,
    'ADS1263_10SPS'  : 0x2,
    'ADS1263_16d6SPS': 0x3,
    'ADS1263_20SPS'  : 0x4,
    'ADS1263_50SPS'  : 0x5,
    'ADS1263_60SPS'  : 0x6,
    'ADS1263_100SPS' : 0x7,
    'ADS1263_400SPS' : 0x8,
    'ADS1263_1200SPS': 0x9,
    'ADS1263_2400SPS': 0xA,
    'ADS1263_4800SPS': 0xB,
    'ADS1263_7200SPS': 0xC,
    'ADS1263_14400SPS': 0xD,
    'ADS1263_19200SPS': 0xE,
    'ADS1263_38400SPS': 0xF,
}

# ADC2 data rate codes (ADC2CFG[7:6])
ADS1263_ADC2_DRATE = {
    'ADS1263_ADC2_10SPS' : 0,
    'ADS1263_ADC2_100SPS': 1,
    'ADS1263_ADC2_400SPS': 2,
    'ADS1263_ADC2_800SPS': 3,
}

ADS1263_ADC2_GAIN = {
    'ADS1263_ADC2_GAIN_1' : 0,
    'ADS1263_ADC2_GAIN_2' : 1,
    'ADS1263_ADC2_GAIN_4' : 2,
    'ADS1263_ADC2_GAIN_8' : 3,
    'ADS1263_ADC2_GAIN_16': 4,
    'ADS1263_ADC2_GAIN_32': 5,
    'ADS1263_ADC2_GAIN_64': 6,
}

# Conversion delay codes (MODE0[3:0])
ADS1263_DELAY = {
    'ADS1263_DELAY_0s'    : 0x0,
    'ADS1263_DELAY_8d7us' : 0x1,
    'ADS1263_DELAY_17us'  : 0x2,
    'ADS1263_DELAY_35us'  : 0x3,
    'ADS1263_DELAY_69us'  : 0x4,
    'ADS1263_DELAY_139us' : 0x5,
    'ADS1263_DELAY_278us' : 0x6,
    'ADS1263_DELAY_555us' : 0x7,
    'ADS1263_DELAY_1d1ms' : 0x8,
    'ADS1263_DELAY_2d2ms' : 0x9,
    'ADS1263_DELAY_4d4ms' : 0xA,
    'ADS1263_DELAY_8d8ms' : 0xB,
    # 0xC..0xF reserved
}

# ADC1 filter type (MODE1[7:5])
ADS1263_FILTER = {
    'SINC1': 0,
    'SINC2': 1,
    'SINC3': 2,
    'SINC4': 3,
    'FIR'  : 4,
    # 5–7 reserved
}

# Chop mode (MODE0[5:4])
ADS1263_CHOP = {
    'CHOP_OFF'        : 0,  # No chop, no IDAC rotation
    'CHOP_ONLY'       : 1,  # Input chop enabled
    'IDAC_ROTATION'   : 2,  # IDAC rotation only
    'CHOP_AND_IDACROT': 3,  # Chop + IDAC rotation
}

# DAC test levels (unchanged from Waveshare semantics)
ADS1263_DAC_VOLT = {
    'ADS1263_DAC_VLOT_4_5'     : 0b01001,
    'ADS1263_DAC_VLOT_3_5'     : 0b01000,
    'ADS1263_DAC_VLOT_3'       : 0b00111,
    'ADS1263_DAC_VLOT_2_75'    : 0b00110,
    'ADS1263_DAC_VLOT_2_625'   : 0b00101,
    'ADS1263_DAC_VLOT_2_5625'  : 0b00100,
    'ADS1263_DAC_VLOT_2_53125' : 0b00011,
    'ADS1263_DAC_VLOT_2_515625': 0b00010,
}

# ---------------------------------------------------------------------------
# Driver class
# ---------------------------------------------------------------------------

class ADS1263:
    """
    ADS1263 driver for Raspberry Pi 5.

    Typical usage (ADC1 differential, 4 pairs scan):

        import ADS1263_pi5 as ADS1263

        adc = ADS1263.ADS1263(vref=5.0)
        if adc.ADS1263_init_ADC1(rate_key='ADS1263_400SPS') < 0:
            raise RuntimeError("ADS1263 init failed")

        adc.ADS1263_SetMode(1)  # differential

        while True:
            codes = adc.ADS1263_GetAll([0, 1, 2, 3])  # 0->AIN0-1, 1->2-3, ...
            volts = [adc.code_to_volts(c) for c in codes]
            print(volts)
            time.sleep(0.01)
    """

    # ------------------------------------------------------------------
    # Construction / low-level helpers
    # ------------------------------------------------------------------
    def __init__(self, vref: float = 5.0):
        self.rst_pin  = config.RST_PIN
        self.cs_pin   = config.CS_PIN
        self.drdy_pin = config.DRDY_PIN

        # 0 = single-ended (AINx vs AINCOM), 1 = differential pairs
        self.ScanMode: int = 1

        # Cached INTERFACE register (status/CRC policy)
        self._interface_reg: int = 0x05  # datasheet reset default

        # Reference voltage used for code -> volts conversion
        self.vref: float = float(vref)

        # Always verify MUX writes (read-back check)
        self._verify_mux_writes: bool = True

    # ---- SPI helpers ----
    def _cs_low(self):
        config.digital_write(self.cs_pin, 0)

    def _cs_high(self):
        config.digital_write(self.cs_pin, 1)

    # ------------------------------------------------------------------
    # Reset / basic SPI primitives
    # ------------------------------------------------------------------
    def ADS1263_reset(self):
        """
        Hardware reset via RESET/PWDN pin.

        Datasheet requires RESET low for at least 2 * tCLK and >0.6 us;
        we use a safe millisecond-scale delay.
        """
        config.digital_write(self.rst_pin, 1)
        config.delay_ms(2)
        config.digital_write(self.rst_pin, 0)
        config.delay_ms(5)
        config.digital_write(self.rst_pin, 1)
        config.delay_ms(5)

    def ADS1263_WriteCmd(self, cmd: int):
        """Send a single-byte command (e.g., START1/STOP1/RDATA1)."""
        self._cs_low()
        config.spi_writebyte([cmd & 0xFF])
        self._cs_high()

    def ADS1263_WriteReg(self, reg: int, value: int):
        """
        Write a single register using WREG sequence.

        WREG opcode:
            first byte: 010r rrrr (0x40 + reg)
            second byte: 000n nnnn, n = number of registers - 1 (here 0)
        """
        self._cs_low()
        config.spi_writebyte([ADS1263_CMD['CMD_WREG'] | (reg & 0x1F), 0x00, value & 0xFF])
        self._cs_high()

    def ADS1263_ReadData(self, reg: int) -> List[int]:
        """
        Read a single register using RREG sequence.

        Returns [value] for compatibility with previous Waveshare-style code.
        """
        self._cs_low()
        config.spi_writebyte([ADS1263_CMD['CMD_RREG'] | (reg & 0x1F), 0x00])
        val = config.spi_readbytes(1)
        self._cs_high()
        return val

    # Small helper to verify MUX writes
    def _write_reg_verified(self, reg: int, value: int, name: str = ""):
        """Write register then read back to verify."""
        self.ADS1263_WriteReg(reg, value)
        rd = self.ADS1263_ReadData(reg)[0]
        if rd != (value & 0xFF):
            label = name or f"reg 0x{reg:02X}"
            print(f"[ADS1263] WARNING: {label} write verify mismatch: wrote 0x{value:02X}, read 0x{rd:02X}")
        return rd

    # ------------------------------------------------------------------
    # Checksum helper (checksum mode, not CRC)
    # ------------------------------------------------------------------
    def ADS1263_CheckSum(self, val: int, byt: int) -> int:
        """
        Checksum as used by Waveshare:
        - Sum all bytes of 'val' (24- or 32-bit word) + 0x9B.
        - If (sum & 0xFF) == byt, returns 0 (OK); else non-zero.
        """
        s = 0
        tmp = val & 0xFFFFFFFF
        while tmp:
            s += (tmp & 0xFF)
            tmp >>= 8
        s += 0x9B
        return ((s & 0xFF) ^ (byt & 0xFF))

    # ------------------------------------------------------------------
    # DRDY wait / ID / INTERFACE
    # ------------------------------------------------------------------
    def ADS1263_WaitDRDY(self, timeout_ms: int = 1000) -> bool:
        """
        Wait for DOUT/DRDY pin to go LOW (new conversion ready).
        Returns True if ready, False on timeout.
        """
        t0 = time.time()
        while config.digital_read(self.drdy_pin) != 0:
            if (time.time() - t0) * 1000.0 > timeout_ms:
                print("[ADS1263] DRDY timeout")
                return False
        return True

    def ADS1263_ReadChipID(self) -> int:
        """
        Read ID register and return ID[7:5]; ADS1262/3 expect 0b001 (0x01).
        """
        id_byte = self.ADS1263_ReadData(ADS1263_REG['REG_ID'])[0]
        return (id_byte >> 5) & 0x07

    def ADS1263_SetInterface(self, timeout_enable: bool = False, status_enable: bool = True,
                             checksum_enable: bool = True):
        """
        Configure INTERFACE register (address 0x02).

        Bits:
            bit3 TIMEOUT: automatic interface timeout (0=disabled, 1=enabled)
            bit2 STATUS : include status byte with conversion data
            bit1:0 CRC  : 00=off, 01=checksum, 10=CRC, 11=reserved

        This driver supports checksum mode (01) for integrity checks.
        CRC mode (10) is not explicitly implemented (checksum is skipped).
        """
        crc_bits = 0b01 if checksum_enable else 0b00
        val = ((1 if timeout_enable else 0) << 3) | ((1 if status_enable else 0) << 2) | crc_bits
        self.ADS1263_WriteReg(ADS1263_REG['REG_INTERFACE'], val)
        self._interface_reg = val

    # ------------------------------------------------------------------
    # ADC1 / ADC2 configuration
    # ------------------------------------------------------------------
    def ADS1263_ConfigADC(self,
                          gain_code: int = ADS1263_GAIN['ADS1263_GAIN_1'],
                          drate_code: int = ADS1263_DRATE['ADS1263_400SPS'],
                          filter_code: int = ADS1263_FILTER['FIR'],
                          chop_code: int = ADS1263_CHOP['CHOP_ONLY'],
                          delay_code: int = ADS1263_DELAY['ADS1263_DELAY_35us'],
                          pga_bypass: bool = True):
        """
        Configure ADC1 (MODE0, MODE1, MODE2, REFMUX).

        MODE2 (0x05):
            bit7    BYPASS  : 1 = bypass PGA, 0 = enable PGA
            bits6:4 GAIN[2:0]
            bits3:0 DR[3:0]

        MODE1 (0x04):
            bits7:5 FILTER[2:0] : SINC1/2/3/4/FIR
            bit4    SBADC
            bit3    SBPOL
            bits2:0 SBMAG

        MODE0 (0x03):
            bit7    REFREV
            bit6    RUNMODE : 0=continuous, 1=pulse
            bits5:4 CHOP[1:0]
            bits3:0 DELAY[3:0]

        REFMUX (0x0F):
            configure reference mux; here we default to AVDD/AVSS.

        Defaults chosen:
            - Continuous conversion
            - FIR filter
            - Chop enabled
            - PGA bypassed (for higher input range; override if needed)
            - AVDD/AVSS reference
        """
        # MODE2: PGA bypass, gain, data rate
        mode2 = ((1 if pga_bypass else 0) << 7) | ((gain_code & 0x7) << 4) | (drate_code & 0xF)
        self.ADS1263_WriteReg(ADS1263_REG['REG_MODE2'], mode2)

        # MODE1: digital filter (sensor bias features left disabled)
        mode1 = ((filter_code & 0x7) << 5)  # SBADC/SBPOL/SBMAG all 0
        self.ADS1263_WriteReg(ADS1263_REG['REG_MODE1'], mode1)

        # MODE0: continuous run, chop, delay
        runmode = 0  # 0 = continuous
        mode0 = ((0 & 0x1) << 7) | ((runmode & 0x1) << 6) | ((chop_code & 0x3) << 4) | (delay_code & 0xF)
        self.ADS1263_WriteReg(ADS1263_REG['REG_MODE0'], mode0)

        # Reference mux: AVDD (RMUXP=100) vs AVSS (RMUXN=011) → 0b10001100 = 0x8C
        # Your previous driver used 0x24 (AVDD/VSS); adjust here if you rely on that.
        refmux = 0x24  # AVDD / AVSS as reference, matching Waveshare behaviour
        self.ADS1263_WriteReg(ADS1263_REG['REG_REFMUX'], refmux)

    def ADS1263_ConfigADC2(self,
                           gain_code: int = ADS1263_ADC2_GAIN['ADS1263_ADC2_GAIN_1'],
                           drate_code: int = ADS1263_ADC2_DRATE['ADS1263_ADC2_400SPS']):
        """
        Configure ADC2 (24-bit auxiliary ADC).

        ADC2CFG (0x15):
            bits7:6 DR2[1:0] : data rate (10, 100, 400, 800 SPS)
            bits5:3 GAIN2[2:0]: gain
            bits2:0 REF2      : reference selection (we pick AVDD/AVSS 0b000)

        MODE0 settings for ADC2 timing still use ADC1's MODE0 delay.
        """
        adc2cfg = ((drate_code & 0x3) << 6) | ((gain_code & 0x7) << 3) | 0x0
        self.ADS1263_WriteReg(ADS1263_REG['REG_ADC2CFG'], adc2cfg)

        # A modest delay for conversions (ADC1 MODE0 still used)
        mode0 = ADS1263_DELAY['ADS1263_DELAY_35us']
        self.ADS1263_WriteReg(ADS1263_REG['REG_MODE0'], mode0)

    # ------------------------------------------------------------------
    # Channel / MUX helpers (ADC1 & ADC2)
    # ------------------------------------------------------------------
    def ADS1263_SetChannal(self, Channel: int):
        """ADC1 single-ended: AINx vs AINCOM (code 0x0A)."""
        if Channel > 10:
            raise ValueError("ADS1263_SetChannal: Channel must be <= 10")
        inpmux = ((Channel & 0xF) << 4) | 0x0A
        if self._verify_mux_writes:
            self._write_reg_verified(ADS1263_REG['REG_INPMUX'], inpmux, name=f"INPMUX(single {Channel})")
        else:
            self.ADS1263_WriteReg(ADS1263_REG['REG_INPMUX'], inpmux)

    def ADS1263_SetChannal_ADC2(self, Channel: int):
        """ADC2 single-ended: AINx vs AINCOM."""
        if Channel > 10:
            raise ValueError("ADS1263_SetChannal_ADC2: Channel must be <= 10")
        inpmux = ((Channel & 0xF) << 4) | 0x0A
        if self._verify_mux_writes:
            self._write_reg_verified(ADS1263_REG['REG_ADC2MUX'], inpmux, name=f"ADC2MUX(single {Channel})")
        else:
            self.ADS1263_WriteReg(ADS1263_REG['REG_ADC2MUX'], inpmux)

    def ADS1263_SetDiffChannal(self, Channel: int):
        """
        ADC1 differential pairs (INPMUX):
            0 -> AIN0 (P) - AIN1 (N)
            1 -> AIN2 (P) - AIN3 (N)
            2 -> AIN4 (P) - AIN5 (N)
            3 -> AIN6 (P) - AIN7 (N)
            4 -> AIN8 (P) - AIN9 (N)
        """
        if Channel == 0:
            inpmux = (0 << 4) | 1
        elif Channel == 1:
            inpmux = (2 << 4) | 3
        elif Channel == 2:
            inpmux = (4 << 4) | 5
        elif Channel == 3:
            inpmux = (6 << 4) | 7
        elif Channel == 4:
            inpmux = (8 << 4) | 9
        else:
            raise ValueError("ADS1263_SetDiffChannal: Channel must be 0..4")

        if self._verify_mux_writes:
            self._write_reg_verified(ADS1263_REG['REG_INPMUX'], inpmux, name=f"INPMUX(diff {Channel})")
        else:
            self.ADS1263_WriteReg(ADS1263_REG['REG_INPMUX'], inpmux)

    def ADS1263_SetDiffChannal_ADC2(self, Channel: int):
        """ADC2 differential pairs with the same mapping as ADC1."""
        if Channel == 0:
            inpmux = (0 << 4) | 1
        elif Channel == 1:
            inpmux = (2 << 4) | 3
        elif Channel == 2:
            inpmux = (4 << 4) | 5
        elif Channel == 3:
            inpmux = (6 << 4) | 7
        elif Channel == 4:
            inpmux = (8 << 4) | 9
        else:
            raise ValueError("ADS1263_SetDiffChannal_ADC2: Channel must be 0..4")

        if self._verify_mux_writes:
            self._write_reg_verified(ADS1263_REG['REG_ADC2MUX'], inpmux, name=f"ADC2MUX(diff {Channel})")
        else:
            self.ADS1263_WriteReg(ADS1263_REG['REG_ADC2MUX'], inpmux)

    def ADS1263_SetMode(self, Mode: int):
        """
        Set scan mode:
            0 => single-ended (use ADS1263_SetChannal / GetChannalValue)
            1 => differential (use ADS1263_SetDiffChannal / GetChannalValue)
        """
        if Mode not in (0, 1):
            raise ValueError("ADS1263_SetMode: Mode must be 0 (single) or 1 (diff)")
        self.ScanMode = Mode

    # ------------------------------------------------------------------
    # ADC1 / ADC2 data read (by command, using INTERFACE)
    # ------------------------------------------------------------------
    def _get_frame_layout(self):
        """
        Compute ADC1 frame layout based on INTERFACE register:

        Returns (has_status, has_checksum, uses_crc)
        """
        reg = self._interface_reg & 0xFF
        has_status = bool(reg & 0x04)
        crc_bits = reg & 0x03
        has_checksum = (crc_bits != 0)
        uses_crc = (crc_bits == 0x02)  # CRC mode vs simple checksum
        return has_status, has_checksum, uses_crc

    def ADS1263_Read_ADC_Data(self) -> int:
        """
        Read ADC1 data via RDATA1 command + INTERFACE-dependent frame.

        Data-byte sequence for ADC1 can be 4, 5, or 6 bytes:
            [STATUS?] DATA0 DATA1 DATA2 DATA3 [CRC/CHK?]

        We honour STATUS/CRC bits in INTERFACE, and perform checksum
        verification when checksum mode is used.
        """
        has_status, has_checksum, uses_crc = self._get_frame_layout()

        # Determine expected byte count
        n_bytes = 4   # 32-bit data
        if has_status:
            n_bytes += 1
        if has_checksum:
            n_bytes += 1

        # Send RDATA1 and read frame
        self._cs_low()
        config.spi_writebyte([ADS1263_CMD['CMD_RDATA1']])
        buf = config.spi_readbytes(n_bytes)
        self._cs_high()

        idx = 0
        status = None
        if has_status:
            status = buf[idx]
            idx += 1

        data_bytes = buf[idx:idx+4]
        idx += 4
        crc_byte = buf[idx] if has_checksum else None

        raw = ((data_bytes[0] << 24) & 0xFF000000) | \
              ((data_bytes[1] << 16) & 0x00FF0000) | \
              ((data_bytes[2] << 8)  & 0x0000FF00) | \
              (data_bytes[3] & 0x000000FF)

        if has_checksum and not uses_crc:
            if self.ADS1263_CheckSum(raw, crc_byte) != 0:
                print("[ADS1263] ADC1 data checksum error")
        # If CRC mode enabled, we do not compute CRC here (not implemented).

        return raw

    def ADS1263_Read_ADC2_Data(self) -> int:
        """
        Read ADC2 data via RDATA2 command + INTERFACE-dependent frame.

        ADC2 frame:
            [STATUS?] DATA0 DATA1 DATA2 PAD(0x00) [CRC/CHK?]

        We parse the 24-bit conversion and return it left-justified in
        a 32-bit integer (same style as Waveshare).
        """
        has_status, has_checksum, uses_crc = self._get_frame_layout()

        # DATA(3) + PAD(1) => 4 bytes base
        n_bytes = 4
        if has_status:
            n_bytes += 1
        if has_checksum:
            n_bytes += 1

        self._cs_low()
        config.spi_writebyte([ADS1263_CMD['CMD_RDATA2']])
        buf = config.spi_readbytes(n_bytes)
        self._cs_high()

        idx = 0
        status = None
        if has_status:
            status = buf[idx]
            idx += 1

        data_bytes = buf[idx:idx+3]
        idx += 3
        pad = buf[idx]  # expected 0x00
        idx += 1
        crc_byte = buf[idx] if has_checksum else None

        raw24 = ((data_bytes[0] << 16) & 0x00FF0000) | \
                ((data_bytes[1] << 8)  & 0x0000FF00) | \
                (data_bytes[2] & 0x000000FF)

        if has_checksum and not uses_crc:
            if self.ADS1263_CheckSum(raw24, crc_byte) != 0:
                print("[ADS1263] ADC2 data checksum error")

        return raw24

    # ------------------------------------------------------------------
    # Channel-level wrappers (single read)
    # ------------------------------------------------------------------
    def ADS1263_GetChannalValue(self, Channel: int) -> int:
        """
        Read ADC1 value for a given channel index, respecting ScanMode.

        IMPORTANT for multi-diff scans:
            After switching the input MUX (single-ended or differential),
            the FIRST conversion still corresponds to the previous MUX
            setting for a while (digital filter settling). We therefore:

                1) Change INPMUX.
                2) Wait for DRDY, then discard the 1st conversion.
                3) Wait for DRDY again and return the 2nd conversion.

        This is the key to avoiding “all channels look like channel 0”
        when scanning across multiple differential pairs.
        """
        # Select the channel/mux
        if self.ScanMode == 0:
            # Single-ended: AINx vs AINCOM
            self.ADS1263_SetChannal(Channel)
        else:
            # Differential pairs: 0..4
            self.ADS1263_SetDiffChannal(Channel)

        # Throw away the first conversion after MUX change
        if not self.ADS1263_WaitDRDY():
            return 0
        _ = self.ADS1263_Read_ADC_Data()

        # Use the second conversion as the valid reading
        if not self.ADS1263_WaitDRDY():
            return 0
        value = self.ADS1263_Read_ADC_Data()
        return value

    def ADS1263_GetChannalValue_ADC2(self, Channel: int) -> int:
        """
        Read ADC2 value for a given channel, respecting ScanMode.

        For ADC2 we trigger a conversion each time by START2 and then
        read the result by command.
        """
        if self.ScanMode == 0:
            self.ADS1263_SetChannal_ADC2(Channel)
        else:
            self.ADS1263_SetDiffChannal_ADC2(Channel)

        self.ADS1263_WriteCmd(ADS1263_CMD['CMD_START2'])

        # For ADC2 we can simply wait for DRDY and read once
        if not self.ADS1263_WaitDRDY():
            return 0
        value = self.ADS1263_Read_ADC2_Data()

        self.ADS1263_WriteCmd(ADS1263_CMD['CMD_STOP2'])
        return value

    # ------------------------------------------------------------------
    # Multi-channel helpers
    # ------------------------------------------------------------------
    def ADS1263_GetAll(self, channels: Sequence[int]) -> List[int]:
        """
        Scan through a list of channel indices and return ADC1 codes.

        - If ScanMode == 0: 'channels' are single-ended AIN indices (0..10).
        - If ScanMode == 1: 'channels' are differential pair indices (0..4)
          mapped as (0-1, 2-3, 4-5, 6-7, 8-9).

        Each channel:
            - Programs INPMUX for that channel.
            - Discards first conversion.
            - Returns the second conversion.

        This implements the correct one-diff-pair-at-a-time behaviour for
        ADS1263 while still giving you a vector of "simultaneous" samples.
        """
        return [self.ADS1263_GetChannalValue(ch) for ch in channels]

    def ADS1263_GetAll_ADC2(self, channels: Sequence[int]) -> List[int]:
        """Scan list of channels using ADC2."""
        return [self.ADS1263_GetChannalValue_ADC2(ch) for ch in channels]

    # ------------------------------------------------------------------
    # Init / shutdown
    # ------------------------------------------------------------------
    def ADS1263_init_ADC1(self,
                          rate_key: str = 'ADS1263_400SPS',
                          gain_key: str = 'ADS1263_GAIN_1',
                          filter_key: str = 'FIR',
                          chop_key: str = 'CHOP_ONLY') -> int:
        """
        Initialize ADC1:

            - Initialize GPIO/SPI via config.module_init().
            - Hardware reset.
            - Verify chip ID.
            - Configure INTERFACE (status + checksum).
            - Configure ADC1 core (MODE0/1/2, REFMUX).
            - START1 continuous conversions.

        Parameters:
            rate_key   : key from ADS1263_DRATE
            gain_key   : key from ADS1263_GAIN
            filter_key : key from ADS1263_FILTER
            chop_key   : key from ADS1263_CHOP
        """
        if config.module_init() != 0:
            print("[ADS1263] module_init() failed")
            return -1

        self.ADS1263_reset()

        chip_id = self.ADS1263_ReadChipID()
        if chip_id != 0x01:
            print(f"[ADS1263] Unexpected chip ID bits: {chip_id:#x}")
            return -1

        # INTERFACE: status + checksum (no timeout by default)
        self.ADS1263_SetInterface(timeout_enable=False, status_enable=True, checksum_enable=True)

        # Stop conversions during configuration
        self.ADS1263_WriteCmd(ADS1263_CMD['CMD_STOP1'])

        # Configure ADC1 core
        self.ADS1263_ConfigADC(
            gain_code=ADS1263_GAIN[gain_key],
            drate_code=ADS1263_DRATE[rate_key],
            filter_code=ADS1263_FILTER[filter_key],
            chop_code=ADS1263_CHOP[chop_key],
            delay_code=ADS1263_DELAY['ADS1263_DELAY_35us'],
            pga_bypass=True,
        )

        # Start continuous conversions on ADC1
        self.ADS1263_WriteCmd(ADS1263_CMD['CMD_START1'])
        return 0

    def ADS1263_init_ADC2(self,
                          rate_key: str = 'ADS1263_ADC2_400SPS',
                          gain_key: str = 'ADS1263_ADC2_GAIN_1') -> int:
        """
        Initialize ADC2 only (assumes ADC1 already initialized or not used).
        """
        # INTERFACE already set by ADC1 init; we leave it as-is.
        self.ADS1263_ConfigADC2(
            gain_code=ADS1263_ADC2_GAIN[gain_key],
            drate_code=ADS1263_ADC2_DRATE[rate_key],
        )
        return 0

    # ------------------------------------------------------------------
    # Utility: convert raw code to volts (ADC1)
    # ------------------------------------------------------------------
    def code_to_volts(self, code: int) -> float:
        """
        Convert 32-bit signed ADC1 code to volts using current vref.
        Assumes bipolar ±FS, where positive full-scale code is 0x7FFFFFFF.
        """
        if code & 0x80000000:
            code = code - 0x100000000
        return (code / 0x7FFFFFFF) * self.vref

    # ------------------------------------------------------------------
    # Optional test helpers (RTD / DAC) — kept for compatibility
    # ------------------------------------------------------------------
    def ADS1263_RTD_Test(self) -> int:
        """
        Example RTD test configuration (kept from Waveshare style, lightly
        modernized). You can adapt this to your own use; not used in normal
        differential multi-scan operation.
        """
        delay = ADS1263_DELAY['ADS1263_DELAY_8d8ms']
        gain  = ADS1263_GAIN['ADS1263_GAIN_1']
        drate = ADS1263_DRATE['ADS1263_20SPS']

        # MODE0: long delay, no chop by default here
        self.ADS1263_WriteReg(ADS1263_REG['REG_MODE0'], delay)

        # Configure IDACMUX / IDACMAG, REFMUX, etc., as needed
        # (left as-is from your original workflow, can be customized)
        # This is intentionally minimal to avoid interfering with your
        # main measurement configuration.
        return 0

    def ADS1263_DAC_Test(self, isPositive: bool, isOpen: bool):
        """
        Simple DAC test: route a test voltage to TDACP/TDACN.
        This does not affect normal ADC operation if unused.
        """
        volt = ADS1263_DAC_VOLT['ADS1263_DAC_VLOT_3']
        reg = ADS1263_REG['REG_TDACP'] if isPositive else ADS1263_REG['REG_TDACN']
        value = (volt | 0x80) if isOpen else 0x00
        self.ADS1263_WriteReg(reg, value)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def ADS1263_Exit(self):
        """Release resources via backend."""
        config.module_exit()