#!/usr/bin/env python3
"""
mycotube_runner.py — MycoTube-only experiment runner (Python controls ESP + ADS1263)

This is a full rewrite (no patching). It matches YOUR ADS1263_pi5.py API exactly.

What it does
- Menu: MycoTube tasks 1,3,4,6
- Asks record duration
- Runs experiment for duration:
  - Every 50 ms (fixed): compute r,i,s + lap + phase, send to ESP ("SET [[r...],i,[s_on,s_off]]")
  - Read ADS1263 diff1..diff4 via ADS1263_GetAll([0,1,2,3]) then code_to_volts()
  - Log CSV per tick with merged LED state
  - Live plot: diff1..diff4, overlay LED state
- After run: prompt for diff to graph (1-4) or Enter to skip
  - Creates summary PNG: chosen diff vs time + proximity(0..1) + red lap markers

Locked behaviors (per your earlier "yes")
- Control tick: 50 ms
- Lap increments on one full spatial cycle (256 steps around ring)
- LED position = circular mean of r
- OFF phase: telemetry continues; intensity forced to 0; position held
- Summary diff y-limits: robust percentile scaling (central 98%)
- Python monotonic time is master
- Ctrl+C / error: STOP ESP, close CSV, optional summary graph only if selected

ESP protocol assumptions (Python is the boss)
- Python -> ESP:
    STOP
    SET [[r1,r2,...],i,[s_on,s_off]]
- ESP -> Python:
    Optional. We read non-blocking and can print with --print-esp, but we do NOT depend on it.

Run
  python3 mycotube_runner.py --esp-port /dev/ttyUSB0 --esp-baud 115200

Notes about your ADS driver
- Uses ADS1263_SetMode(1) for differential scan mode
- Reads all 4 diffs with ADS1263_GetAll([0,1,2,3]) then code_to_volts()
- ADS driver handles DRDY wait + first-sample discard per channel internally
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------- Optional imports ----------
HAVE_SERIAL = False
HAVE_ADS = False
HAVE_MPL = False

try:
    import serial  # pyserial
    HAVE_SERIAL = True
except Exception:
    HAVE_SERIAL = False

try:
    import ADS1263_pi5 as ADS
    HAVE_ADS = True
except Exception:
    HAVE_ADS = False

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# =========================
# Utilities
# =========================

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def circ_dist_256(a: float, b: float) -> float:
    d = (a - b) % 256.0
    return min(d, 256.0 - d)

def circular_mean_256(xs: List[int]) -> float:
    """Circular mean for positions on a 0..255 ring."""
    if not xs:
        return float("nan")
    angs = [2.0 * math.pi * (x % 256) / 256.0 for x in xs]
    s = sum(math.sin(a) for a in angs)
    c = sum(math.cos(a) for a in angs)
    if s == 0.0 and c == 0.0:
        return float(xs[0] % 256)
    ang = math.atan2(s, c)
    if ang < 0:
        ang += 2.0 * math.pi
    return (ang / (2.0 * math.pi)) * 256.0

def robust_ylim(y: List[float], qlo: float = 0.01, qhi: float = 0.99) -> Tuple[float, float]:
    y = [v for v in y if not (math.isnan(v) or math.isinf(v))]
    if not y:
        return (-1.0, 1.0)
    ys = sorted(y)
    n = len(ys)
    if n < 10:
        mn, mx = min(ys), max(ys)
        if mn == mx:
            return (mn - 1e-6, mx + 1e-6)
        pad = 0.05 * (mx - mn)
        return (mn - pad, mx + pad)
    lo_i = int(clamp(qlo * (n - 1), 0, n - 1))
    hi_i = int(clamp(qhi * (n - 1), 0, n - 1))
    lo, hi = ys[lo_i], ys[hi_i]
    if lo == hi:
        lo -= 1e-6
        hi += 1e-6
    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)


# =========================
# Models
# =========================

@dataclass
class LEDState:
    t_s: float
    phase: str           # "ON"/"OFF"
    lap: int
    x_center: float      # 0..255 (held during OFF)
    width: int
    intensity: int       # 0..100

@dataclass
class SampleRow:
    t_s: float
    diff1_V: float
    diff2_V: float
    diff3_V: float
    diff4_V: float
    led_phase: str
    led_lap: int
    led_x_center_256: float
    led_width: int
    led_intensity: int


# =========================
# ESP serial link
# =========================

class ESPLink:
    def __init__(self, port: str, baud: int) -> None:
        if not HAVE_SERIAL:
            raise RuntimeError("pyserial not installed. Install: pip install pyserial")
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.0)
        self.buf = bytearray()

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass

    def write_line(self, s: str) -> None:
        if not s.endswith("\n"):
            s += "\n"
        self.ser.write(s.encode("utf-8", errors="replace"))

    def read_lines(self, max_lines: int = 200) -> List[str]:
        lines: List[str] = []
        try:
            n = self.ser.in_waiting
        except Exception:
            n = 0
        if n:
            self.buf.extend(self.ser.read(n))
        while len(lines) < max_lines:
            i = self.buf.find(b"\n")
            if i < 0:
                break
            raw = self.buf[:i]
            del self.buf[:i + 1]
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                lines.append(line)
        return lines

def format_set_payload(r: List[int], intensity: int, s_on: float, s_off: float) -> str:
    r_str = ",".join(str(int(x) % 256) for x in r)
    return f"SET [[{r_str}],{int(intensity)},[{float(s_on):.6f},{float(s_off):.6f}]]"


# =========================
# ADS1263 reader (YOUR driver API)
# =========================

class ADSReader:
    """
    Uses your ADS1263_pi5.py exactly:
      adc = ADS.ADS1263(vref=2.5)
      adc.ADS1263_init_ADC1(rate_key="ADS1263_400SPS")
      adc.ADS1263_SetMode(1)  # differential
      codes = adc.ADS1263_GetAll([0,1,2,3])
      volts = [adc.code_to_volts(c) for c in codes]
    """

    def __init__(self, vref: float, rate_key: str) -> None:
        if not HAVE_ADS:
            raise RuntimeError("ADS1263_pi5 not found. Run on RP5 environment.")

        self.adc = ADS.ADS1263(vref=vref)

        rc = self.adc.ADS1263_init_ADC1(rate_key=rate_key)
        if rc < 0:
            raise RuntimeError("ADS1263_init_ADC1 failed")

        # Differential scan mode (per your driver)
        self.adc.ADS1263_SetMode(1)

        # diff indices: 0->0-1, 1->2-3, 2->4-5, 3->6-7
        self.channels = [0, 1, 2, 3]

        # Fail-fast sanity scan
        codes = self.adc.ADS1263_GetAll(self.channels)
        if not isinstance(codes, list) or len(codes) != 4:
            raise RuntimeError(f"ADS1263_GetAll returned {codes!r} (expected list of 4)")

    def read_diffs(self) -> Tuple[float, float, float, float]:
        codes = self.adc.ADS1263_GetAll(self.channels)
        v = [self.adc.code_to_volts(int(c)) for c in codes]
        return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))


# =========================
# Task system (MycoTube only)
# =========================

DIFF_CLOSE_X = {1: 32, 2: 96, 3: 160, 4: 224}

def wall_range(wall: int) -> Tuple[int, int]:
    if wall not in (1, 2, 3, 4):
        raise ValueError("wall must be 1..4")
    start = (wall - 1) * 64
    return start, start + 63

def bar_frame(start_x: int, width: int) -> List[int]:
    start_x %= 256
    return [((start_x + k) % 256) for k in range(int(width))]

def wall_bar_frame(walls: List[int], local_x: int, width: int) -> List[int]:
    out: List[int] = []
    for w in walls:
        base = (w - 1) * 64
        for k in range(int(width)):
            out.append((base + ((local_x + k) % 64)) % 256)
    seen = set()
    uniq: List[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

def panel_combos(mode: int) -> List[List[int]]:
    if mode == 1:
        return [[1], [2], [3], [4]]
    if mode == 2:
        return [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
    if mode == 3:
        return [[1,2,3],[1,2,4],[1,3,4],[2,3,4]]
    if mode == 4:
        return [[1,2,3,4]]
    return [[1],[2],[3],[4]]

def speed_cps(speed: str) -> float:
    # locked: slow=1, medium=4, fast=10 columns/sec
    return {"slow": 1.0, "medium": 4.0, "fast": 10.0}.get(speed, 4.0)

def resolve_task() -> Dict[str, Any]:
    while True:
        s = input(
            "\nSelect MycoTube task:\n"
            "  1) Task 1 — Brightness ramp (single wall)\n"
            "  3) Task 3 — Flashing light bar\n"
            "  4) Task 4 — Rotating light bar\n"
            "  6) Task 6 — Pattern regeneration\n"
            "Enter task number (or q to quit): "
        ).strip().lower()

        if s in ("q", "quit", "exit"):
            return {"task": "quit"}
        if s not in ("1", "3", "4", "6"):
            print("Invalid selection.")
            continue

        t = int(s)
        cfg: Dict[str, Any] = {"task": t}

        if t == 1:
            cfg["wall"] = int(input("Wall/panel (1-4): ").strip())
            cfg["step_s"] = 3.0
            cfg["i_min"] = 0
            cfg["i_max"] = 100

        elif t == 3:
            cfg["width"] = int(input("Bar width (1/3/5/7): ").strip())
            cfg["on_s"] = float(input("On-duration seconds (10/5/2/1): ").strip())
            cfg["off_s"] = cfg["on_s"]  # locked default
            cfg["mode"] = int(input("Panel-combo mode (1=singles,2=pairs,3=triples,4=all): ").strip())
            cfg["local_x"] = 32

        elif t == 4:
            cfg["width"] = int(input("Bar width (1/3/5/7): ").strip())
            cfg["bars"] = int(input("Bars at once (1-5): ").strip())
            spd = input("Speed (slow/medium/fast): ").strip().lower()
            cfg["speed"] = spd if spd in ("slow", "medium", "fast") else "medium"

        elif t == 6:
            kind = input("Pattern type (triangle/diminishing-bars): ").strip().lower()
            cfg["kind"] = kind if kind in ("triangle", "diminishing-bars") else "triangle"
            spd = input("Speed (slow/medium/fast): ").strip().lower()
            cfg["speed"] = spd if spd in ("slow", "medium", "fast") else "medium"

        return cfg

def compute_led_command(cfg: Dict[str, Any], t_run: float, last_on_x: float) -> Tuple[List[int], int, float, float, int, str, float, float]:
    """
    Returns:
      r, intensity, s_on, s_off, lap, phase, x_center, last_on_x_updated

    Lap rule:
    - Task 4 and 6: lap increments after full 256-position spatial cycle.
    - Task 3: lap increments after each ON+OFF cycle (because it is not spatially moving).
    - Task 1: lap=0 (not meaningful).
    """
    task = int(cfg["task"])

    r: List[int] = []
    intensity = 0
    s_on = 999999.0
    s_off = 0.0
    lap = 0
    phase = "ON"
    x_center = last_on_x

    if task == 1:
        a, b = wall_range(int(cfg["wall"]))
        r = list(range(a, b + 1))
        step_s = float(cfg["step_s"])
        k = int(t_run // step_s)
        intensity = int(clamp(int(cfg["i_min"]) + k, cfg["i_min"], cfg["i_max"]))
        phase = "ON"
        lap = 0

    elif task == 3:
        width = int(cfg["width"])
        on_s = float(cfg["on_s"])
        off_s = float(cfg["off_s"])
        s_on, s_off = on_s, off_s

        cycle = on_s + off_s
        lap = int(t_run // cycle)
        t_in = t_run - lap * cycle
        phase = "ON" if t_in < on_s else "OFF"

        combos = panel_combos(int(cfg["mode"]))
        active = combos[lap % len(combos)]

        if phase == "ON":
            r = wall_bar_frame(active, int(cfg["local_x"]) % 64, width)
            intensity = 100
        else:
            r = []
            intensity = 0

    elif task == 4:
        width = int(cfg["width"])
        bars = int(cfg["bars"])
        cps = speed_cps(str(cfg["speed"]))

        x0 = int((t_run * cps) % 256)
        lap = int((t_run * cps) // 256)
        phase = "ON"
        intensity = 100

        centers = [(x0 + int(round(k * 256 / bars))) % 256 for k in range(bars)]
        raw: List[int] = []
        for c in centers:
            raw.extend(bar_frame(c, width))
        seen = set()
        r = []
        for p in raw:
            if p not in seen:
                seen.add(p)
                r.append(p)

    elif task == 6:
        cps = speed_cps(str(cfg["speed"]))
        x0 = int((t_run * cps) % 256)
        lap = int((t_run * cps) // 256)
        phase = "ON"

        if str(cfg["kind"]) == "triangle":
            step = int((t_run * cps) % 14)
            w = step + 1 if step < 7 else (14 - step)
            w = int(clamp(w, 1, 7))
            r = bar_frame(x0, w)
            intensity = 100
        else:
            intensities = [100, 85, 70, 55, 40, 25, 10]
            idx = int((t_run * 20.0) % 7)  # 20 Hz internal multiplex
            r = [((x0 + idx) % 256)]
            intensity = intensities[idx]

    # Position + OFF holding (locked)
    if r:
        x_center = circular_mean_256(r)
        last_on_x = x_center
    else:
        x_center = last_on_x

    return r, intensity, s_on, s_off, lap, phase, x_center, last_on_x


# =========================
# Plotting
# =========================

class LivePlot:
    def __init__(self, window_s: float) -> None:
        if not HAVE_MPL:
            raise RuntimeError("matplotlib not installed.")
        self.window_s = float(window_s)

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.l1, = self.ax.plot([], [], label="diff1")
        self.l2, = self.ax.plot([], [], label="diff2")
        self.l3, = self.ax.plot([], [], label="diff3")
        self.l4, = self.ax.plot([], [], label="diff4")
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("differential voltage (V)")
        self.ax.grid(True)
        self.ax.legend(loc="upper right")
        self.txt = self.ax.text(0.01, 0.99, "", transform=self.ax.transAxes,
                                ha="left", va="top", fontsize=10)

    def update(self, t: List[float], d1: List[float], d2: List[float], d3: List[float], d4: List[float], led: LEDState) -> None:
        if not t:
            return

        if self.window_s > 0:
            cutoff = t[-1] - self.window_s
            idx = 0
            while idx < len(t) and t[idx] < cutoff:
                idx += 1
            if idx > 0:
                del t[:idx]
                del d1[:idx]
                del d2[:idx]
                del d3[:idx]
                del d4[:idx]

        self.l1.set_data(t, d1)
        self.l2.set_data(t, d2)
        self.l3.set_data(t, d3)
        self.l4.set_data(t, d4)

        self.ax.set_xlim(max(0.0, t[-1] - max(self.window_s, 60.0)), t[-1] + 0.25)
        yall = d1 + d2 + d3 + d4
        ylo, yhi = robust_ylim(yall, 0.01, 0.99)
        self.ax.set_ylim(ylo, yhi)

        self.txt.set_text(f"LED: {led.phase} | lap {led.lap} | x {led.x_center:.2f} | width {led.width} | i {led.intensity}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

def postrun_graph(csv_path: str, diff_idx: int, out_png: str) -> None:
    if not HAVE_MPL:
        raise RuntimeError("matplotlib not installed.")

    t: List[float] = []
    dv: List[float] = []
    lap: List[int] = []
    x: List[float] = []

    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t.append(float(row["t_s"]))
                dv.append(float(row[f"diff{diff_idx}_V"]))
                lap.append(int(row["led_lap"]))
                x.append(float(row["led_x_center_256"]))
            except Exception:
                continue

    if not t:
        raise RuntimeError("No data in CSV.")

    lap_times: List[float] = []
    last = lap[0]
    for i in range(1, len(lap)):
        if lap[i] > last:
            lap_times.append(t[i])
            last = lap[i]

    x_close = DIFF_CLOSE_X[diff_idx]
    prox = [1.0 - (circ_dist_256(xi, x_close) / 128.0) for xi in x]

    plt.ioff()
    fig, ax1 = plt.subplots()
    ax1.plot(t, dv, label=f"diff{diff_idx} (V)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("differential voltage (V)")
    ax1.grid(True)
    ylo, yhi = robust_ylim(dv, 0.01, 0.99)
    ax1.set_ylim(ylo, yhi)

    for ts in lap_times:
        ax1.axvline(ts, color="red", linewidth=1.0)

    ax2 = ax1.twinx()
    ax2.plot(t, prox, label="light proximity (0-1)")
    ax2.set_ylabel("light proximity (0–1)")
    ax2.set_ylim(-0.05, 1.05)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right")

    ax1.set_title(f"diff{diff_idx} with lap markers + proximity (x_close={x_close})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# =========================
# Runner
# =========================

class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop_requested = False
        self.esp: Optional[ESPLink] = None
        self.adc: Optional[ADSReader] = None

        signal.signal(signal.SIGINT, self._on_sig)
        signal.signal(signal.SIGTERM, self._on_sig)

    def _on_sig(self, *_: Any) -> None:
        self.stop_requested = True

    def connect(self) -> None:
        if self.args.esp_port:
            self.esp = ESPLink(self.args.esp_port, self.args.esp_baud)
            self.stop_esp()
        else:
            print("WARNING: no ESP port provided; LED control disabled.")

        if self.args.no_adc:
            self.adc = None
            print("ADC disabled (--no-adc).")
        else:
            self.adc = ADSReader(vref=self.args.vref, rate_key=self.args.rate_key)

    def stop_esp(self) -> None:
        if self.esp is None:
            return
        try:
            self.esp.write_line("STOP")
        except Exception:
            pass

    def close(self) -> None:
        self.stop_esp()
        if self.esp is not None:
            self.esp.close()

    def run(self) -> None:
        os.makedirs(self.args.runs_dir, exist_ok=True)

        while not self.stop_requested:
            cfg = resolve_task()
            if cfg.get("task") == "quit":
                return

            try:
                duration = float(input("Record duration (s): ").strip())
                if duration <= 0:
                    raise ValueError
            except Exception:
                print("Invalid duration.")
                continue

            csv_path = self.run_one(cfg, duration)
            if not csv_path:
                continue

            sel = input("Graph a diff channel? [1-4, Enter=skip]: ").strip()
            if sel == "":
                continue
            try:
                di = int(sel)
                if di not in (1, 2, 3, 4):
                    raise ValueError
            except Exception:
                print("Invalid; skipping graph.")
                continue

            out_png = os.path.splitext(csv_path)[0] + f"_diff{di}.png"
            try:
                postrun_graph(csv_path, di, out_png)
                print(f"Saved graph: {out_png}")
            except Exception as e:
                print(f"Graph failed: {e}")

    def run_one(self, cfg: Dict[str, Any], duration: float) -> Optional[str]:
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = f"task{cfg['task']}"
        if int(cfg["task"]) == 1:
            tag += f"_wall{cfg['wall']}"
        base = f"run_{ts}_{tag}"
        csv_path = os.path.join(self.args.runs_dir, base + ".csv")

        dt = 0.050  # locked control tick
        t0 = time.monotonic()
        next_tick = t0

        last_on_x = 0.0
        led = LEDState(t_s=0.0, phase="OFF", lap=0, x_center=0.0, width=0, intensity=0)

        t_buf: List[float] = []
        d1_buf: List[float] = []
        d2_buf: List[float] = []
        d3_buf: List[float] = []
        d4_buf: List[float] = []

        live: Optional[LivePlot] = None
        if not self.args.no_plot and HAVE_MPL:
            live = LivePlot(window_s=self.args.plot_window_s)
        elif not self.args.no_plot and not HAVE_MPL:
            print("WARNING: matplotlib missing; live plot disabled.")

        self.stop_esp()

        print(f"Running {base} for {duration:.1f}s")
        print(f"Saving CSV to: {csv_path}")

        fields = [
            "t_s",
            "diff1_V", "diff2_V", "diff3_V", "diff4_V",
            "led_phase", "led_lap", "led_x_center_256", "led_width", "led_intensity",
        ]

        try:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()

                next_plot = t0
                plot_dt = 0.10  # 10 Hz plotting

                while True:
                    if self.stop_requested:
                        raise KeyboardInterrupt

                    now = time.monotonic()
                    if now < next_tick:
                        if self.esp is not None:
                            lines = self.esp.read_lines(200)
                            if self.args.print_esp:
                                for ln in lines:
                                    print(f"[ESP] {ln}")
                        time.sleep(min(0.005, next_tick - now))
                        continue

                    t_s = now - t0
                    if t_s >= duration:
                        break

                    # Compute LED command at this tick
                    r_list, inten, s_on, s_off, lap, phase, x_center, last_on_x = compute_led_command(cfg, t_s, last_on_x)

                    # Locked OFF behavior already applied in compute_led_command
                    led = LEDState(
                        t_s=t_s,
                        phase=phase,
                        lap=int(lap),
                        x_center=float(x_center),
                        width=len(r_list),
                        intensity=int(inten),
                    )

                    # Send to ESP
                    if self.esp is not None:
                        try:
                            self.esp.write_line(format_set_payload(r_list, inten, s_on, s_off))
                        except Exception:
                            pass

                    # Read ADC
                    if self.adc is None:
                        d1 = d2 = d3 = d4 = 0.0
                    else:
                        d1, d2, d3, d4 = self.adc.read_diffs()

                    # Log CSV row
                    row = SampleRow(
                        t_s=t_s,
                        diff1_V=float(d1),
                        diff2_V=float(d2),
                        diff3_V=float(d3),
                        diff4_V=float(d4),
                        led_phase=led.phase,
                        led_lap=led.lap,
                        led_x_center_256=led.x_center,
                        led_width=led.width,
                        led_intensity=led.intensity,
                    )

                    w.writerow({
                        "t_s": f"{row.t_s:.6f}",
                        "diff1_V": f"{row.diff1_V:+.9f}",
                        "diff2_V": f"{row.diff2_V:+.9f}",
                        "diff3_V": f"{row.diff3_V:+.9f}",
                        "diff4_V": f"{row.diff4_V:+.9f}",
                        "led_phase": row.led_phase,
                        "led_lap": str(row.led_lap),
                        "led_x_center_256": f"{row.led_x_center_256:.6f}",
                        "led_width": str(row.led_width),
                        "led_intensity": str(row.led_intensity),
                    })

                    # Update live plot buffers
                    t_buf.append(t_s)
                    d1_buf.append(float(d1))
                    d2_buf.append(float(d2))
                    d3_buf.append(float(d3))
                    d4_buf.append(float(d4))

                    if live is not None and now >= next_plot:
                        live.update(t_buf, d1_buf, d2_buf, d3_buf, d4_buf, led)
                        next_plot = now + plot_dt

                    next_tick += dt

        except KeyboardInterrupt:
            print("Aborted (Ctrl+C).")
            self.stop_esp()
            return csv_path if os.path.exists(csv_path) else None
        except Exception as e:
            print(f"Run failed: {e}")
            self.stop_esp()
            return csv_path if os.path.exists(csv_path) else None
        finally:
            self.stop_esp()

        print("Run complete.")
        return csv_path


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--esp-port", default="", help="ESP serial port (e.g., /dev/ttyUSB0)")
    p.add_argument("--esp-baud", type=int, default=115200)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot-window-s", type=float, default=300.0)
    p.add_argument("--print-esp", action="store_true", help="Print ESP lines to terminal (debug).")

    p.add_argument("--no-adc", action="store_true")
    p.add_argument("--vref", type=float, default=2.5)
    p.add_argument("--rate-key", default="ADS1263_400SPS")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    r = Runner(args)
    try:
        r.connect()
        r.run()
    finally:
        r.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())