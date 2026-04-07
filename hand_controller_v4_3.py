

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import urllib.request
import os
import traceback
import random
import math
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Fix Windows PowerShell unicode encoding crash
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


HAS_VOLUME  = False
volume_ctrl = None
VOL_MIN = VOL_MAX = 0

try:
    import ctypes
    _comtypes_mod  = importlib.import_module("comtypes")
    _pycaw_mod     = importlib.import_module("pycaw.pycaw")

    CLSCTX_ALL          = _comtypes_mod.CLSCTX_ALL                         # type: ignore[attr-defined]
    AudioUtilities      = _pycaw_mod.AudioUtilities                         # type: ignore[attr-defined]
    IAudioEndpointVolume= _pycaw_mod.IAudioEndpointVolume                   # type: ignore[attr-defined]

    _devices    = AudioUtilities.GetSpeakers()
    _interface  = _devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_ctrl = ctypes.cast(_interface, ctypes.POINTER(IAudioEndpointVolume))
    VOL_MIN, VOL_MAX = volume_ctrl.GetVolumeRange()[:2]
    HAS_VOLUME  = True
    print("[OK] pycaw volume control enabled")
except ImportError:
    print("[INFO] pycaw/comtypes not installed - using keyboard volume fallback")
except Exception as e:
    print(f"[INFO] Volume control unavailable: {e}")

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0


#  CONFIG

CAMERA_INDEX        = 0
MODEL_FILE          = "hand_landmarker.task"
SCREEN_W, SCREEN_H  = pyautogui.size()

MODES = ["CURSOR", "VOLUME", "ZOOM", "PRESENT"]

THEME = {
    "CURSOR":  {"primary": (0, 255, 180),  "accent": (180, 0, 255)},
    "VOLUME":  {"primary": (0, 200, 255),  "accent": (255, 150, 0)},
    "ZOOM":    {"primary": (255, 220, 0),  "accent": (0, 180, 255)},
    "PRESENT": {"primary": (255, 80, 120), "accent": (255, 200, 0)},
}

# ── Thresholds (normalised to hand scale)
T = {
    "PINCH_ON":    0.13,   # slightly easier to trigger pinch
    "PINCH_OFF":   0.20,   # wider hysteresis = cleaner release
    "RMID_ON":     0.13,
    "RMID_OFF":    0.20,
    "DRAG_ON":     0.09,
    "DRAG_OFF":    0.15,
    "CURL_UP":     0.60,
    "CURL_DOWN":   0.35,
    "SCROLL_DEAD": 0.008,  # smaller dead zone = scroll responds sooner
    "SCROLL_SPEED":20000.0,   # pixels of scroll per pixel of hand movement
    "SWIPE_MIN":   0.18,   # slightly easier swipe trigger
}

# ── Gesture buffer config: {gesture: (buffer_size, confidence_ratio)} ─────
# Smaller buffers = faster response. Lower ratio = more forgiving detection.
BUFFER_CFG = {
    "MOVE_CURSOR":  (2,  0.50),   # was (3,0.67) — instant cursor response
    "LEFT_CLICK":   (4,  0.75),   # was (5,0.80) — slightly faster click
    "RIGHT_CLICK":  (5,  0.80),   # was (6,0.83)
    "DRAG":         (3,  0.67),   # was (4,0.75)
    "SCROLL":       (2,  0.50),   # was (4,0.75) — scroll starts immediately
    "VOLUME":       (2,  0.50),   # was (3,0.67)
    "ZOOM":         (2,  0.50),
    "PAUSE":        (4,  0.75),
    "MODE_SWITCH":  (36, 1.00),   # keep — intentional hold required
    "SWIPE_LEFT":   (5,  0.80),
    "SWIPE_RIGHT":  (5,  0.80),
    "IDLE":         (2,  0.50),
}

COOLDOWNS = {
    "LEFT_CLICK":  0.35,   # was 0.40 — snappier clicking
    "RIGHT_CLICK": 0.45,   # was 0.50
    "MODE_SWITCH": 2.00,
    "SWIPE_LEFT":  0.50,   # was 0.60
    "SWIPE_RIGHT": 0.50,
}

# Lower number = higher priority; only one gesture fires per frame
PRIORITY = {
    "PAUSE":       1,
    "MODE_SWITCH": 2,
    "LEFT_CLICK":  3,
    "RIGHT_CLICK": 4,
    "DRAG":        5,
    "SCROLL":      6,
    "ZOOM":        6,
    "VOLUME":      6,
    "SWIPE_LEFT":  7,
    "SWIPE_RIGHT": 7,
    "MOVE_CURSOR": 8,
    "IDLE":        9,
}

# Gestures allowed in each mode (easy extensibility — add here + BUFFER_CFG)
MODE_GESTURES = {
    "CURSOR":  {"MOVE_CURSOR", "LEFT_CLICK", "RIGHT_CLICK", "DRAG", "SCROLL",
                "PAUSE", "MODE_SWITCH"},
    "VOLUME":  {"VOLUME", "PAUSE", "MODE_SWITCH"},
    "ZOOM":    {"ZOOM", "PAUSE", "MODE_SWITCH"},
    "PRESENT": {"MOVE_CURSOR", "LEFT_CLICK", "SWIPE_LEFT", "SWIPE_RIGHT",
                "PAUSE", "MODE_SWITCH"},
}


#  MATH HELPERS

def edist(a, b):
    """Euclidean distance between two MediaPipe NormalizedLandmarks."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

def hand_scale(lm):
    """Wrist-to-middle-MCP distance; normalises all thresholds to hand size."""
    return edist(lm[0], lm[9]) + 1e-6

def ndist(a, b, hs):
    """Normalised distance (divided by hand scale)."""
    return edist(a, b) / hs

def fingers_state(lm):
    """
    Returns (up: list[bool], hs: float).
    up[0]=thumb, up[1]=index, up[2]=middle, up[3]=ring, up[4]=pinky.
    Thumb rule: tip.x < MCP.x works because the frame is mirrored.
    Others: tip.y < PIP.y (tip above PIP joint in image coordinates).
    """
    hs = hand_scale(lm)
    up = [
        lm[4].x  < lm[3].x,   # thumb
        lm[8].y  < lm[6].y,   # index
        lm[12].y < lm[10].y,  # middle
        lm[16].y < lm[14].y,  # ring
        lm[20].y < lm[18].y,  # pinky
    ]
    return up, hs

#  GESTURE BUFFER — debounce / stability

class GestureBuffer:
    """
    Sliding window debounce.  Each frame we push which raw gestures are
    present; a gesture only fires when its deque is full and the True ratio
    >= its required confidence.  MODE_SWITCH requires 100% ratio (36/36).
    """
    def __init__(self):     
        self.buffers = {g: deque(maxlen=BUFFER_CFG[g][0]) for g in BUFFER_CFG}

    def push(self, raw_gestures):
        """raw_gestures: list of candidate gesture names detected this frame."""
        for g in self.buffers:
            self.buffers[g].append(g in raw_gestures)

    def confirmed(self, mode):
        """Return highest-priority confirmed gesture for the current mode."""
        allowed = MODE_GESTURES[mode]
        results = []
        for g, buf in self.buffers.items():
            if g not in allowed:
                continue
            size, ratio = BUFFER_CFG[g]
            if len(buf) == size and sum(buf) / size >= ratio:
                results.append(g)
        if not results:
            return "IDLE"
        return min(results, key=lambda g: PRIORITY[g])

#  CURSOR SMOOTHER — exponential moving average

class CursorSmoother:
    """
    Dual-speed EMA: fast alpha when hand moves quickly, slow alpha when still.
    This gives snappy large movements AND steady fine positioning.
    """
    def __init__(self, alpha_slow=0.25, alpha_fast=0.65, speed_threshold=80):
        self.alpha_slow      = alpha_slow       # smooth when nearly still
        self.alpha_fast      = alpha_fast       # responsive when moving fast
        self.speed_threshold = speed_threshold  # pixels/frame to switch modes
        self.sx = SCREEN_W / 2
        self.sy = SCREEN_H / 2

    def update(self, rx, ry):
        dist = math.sqrt((rx - self.sx) ** 2 + (ry - self.sy) ** 2)
        # blend alpha based on movement speed
        t     = min(dist / self.speed_threshold, 1.0)
        alpha = self.alpha_slow + t * (self.alpha_fast - self.alpha_slow)
        self.sx = alpha * rx + (1 - alpha) * self.sx
        self.sy = alpha * ry + (1 - alpha) * self.sy
        return int(self.sx), int(self.sy)


#  VISUAL EFFECTS

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]
FINGERTIPS = [4, 8, 12, 16, 20]


class Particle:
    def __init__(self, x, y, color):
        self.x     = x + random.randint(-8, 8)
        self.y     = y + random.randint(-8, 8)
        self.vx    = random.uniform(-2.5, 2.5)
        self.vy    = random.uniform(-4, -0.5)
        self.life  = 1.0
        self.decay = random.uniform(0.035, 0.08)
        self.size  = random.randint(2, 6)
        self.color = color

    def update(self):
        self.vy  += 0.1
        self.x   += self.vx
        self.y   += self.vy
        self.life -= self.decay
        return self.life > 0

    def draw(self, frame):
        c = tuple(int(v * self.life) for v in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)),
                   max(1, int(self.size * self.life)), c, -1)


class Ripple:
    def __init__(self, x, y, color):
        self.x, self.y, self.r, self.color = x, y, 5, color
        self.max_r = 70
        self.life  = 1.0

    def update(self):
        self.r    += 5
        self.life  = max(0, 1 - self.r / self.max_r)
        return self.r < self.max_r

    def draw(self, frame):
        c = tuple(int(v * self.life) for v in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), int(self.r), c, 2)
        cv2.circle(frame, (int(self.x), int(self.y)),
                   max(1, int(self.r * 0.6)), c, 1)


class TrailSystem:
    def __init__(self, maxlen=30, lifetime=0.4):
        self.pts = deque(maxlen=maxlen)
        self.lt  = lifetime

    def push(self, x, y):
        self.pts.append((x, y, time.time()))

    def draw(self, frame, color):
        now   = time.time()
        valid = [(x, y, t) for x, y, t in self.pts if now - t < self.lt]
        for k in range(1, len(valid)):
            a = (k - 1) / max(len(valid) - 1, 1)
            c = tuple(int(v * a) for v in color)
            cv2.line(frame,
                     (valid[k-1][0], valid[k-1][1]),
                     (valid[k][0],   valid[k][1]),
                     c, max(1, int(4 * a)))


def glow_circle(frame, cx, cy, r, color, layers=4):
    for i in range(layers, 0, -1):
        ov = frame.copy()
        cv2.circle(ov, (cx, cy), r + (layers - i) * 4, color, -1)
        cv2.addWeighted(ov, 0.12 * i / layers, frame,
                        1 - 0.12 * i / layers, 0, frame)


def glow_line(frame, p1, p2, color, thick=2):
    ov = frame.copy()
    cv2.line(ov, p1, p2, color, thick + 6)
    cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
    cv2.line(frame, p1, p2, color, thick)


def draw_hand(frame, lm, w, h, primary, accent):
    for a, b in CONNECTIONS:
        glow_line(frame,
                  (int(lm[a].x * w), int(lm[a].y * h)),
                  (int(lm[b].x * w), int(lm[b].y * h)),
                  primary, 1)
    for i, p in enumerate(lm):
        x, y = int(p.x * w), int(p.y * h)
        if i in FINGERTIPS:
            glow_circle(frame, x, y, 9, accent, 5)
            cv2.circle(frame, (x, y), 7, accent, -1)
            cv2.circle(frame, (x, y), 9, (255, 255, 255), 1)
        elif i == 0:
            glow_circle(frame, x, y, 7, primary, 3)
            cv2.circle(frame, (x, y), 5, primary, -1)
        else:
            cv2.circle(frame, (x, y), 4, primary, -1)
            cv2.circle(frame, (x, y), 5, (200, 200, 200), 1)

def draw_hud(frame, mode, gesture, fps, w, h, primary, extra=""):
    # top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 58), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    # mode badge
    cv2.rectangle(frame, (8, 8), (130, 50), primary, -1)
    cv2.putText(frame, mode, (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    # gesture colour
    GCOL = {
        "LEFT_CLICK":  (0, 255, 255),  "RIGHT_CLICK": (80, 80, 255),
        "SCROLL":      (0, 180, 255),  "DRAG":        (255, 140, 0),
        "VOLUME":      (0, 220, 255),  "ZOOM":        (255, 220, 0),
        "PAUSE":       (200, 200, 0),  "MODE_SWITCH": (255, 60, 60),
        "SWIPE_LEFT":  (255, 100, 180),"SWIPE_RIGHT": (100, 255, 180),
        "IDLE":        (70, 70, 70),
    }
    gc = GCOL.get(gesture, (220, 220, 220))
    cv2.putText(frame, gesture, (145, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, gc, 2, cv2.LINE_AA)
    # fps
    fc = (0, 255, 100) if fps > 20 else (0, 120, 255)
    cv2.putText(frame, f"{fps}fps", (w - 90, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fc, 2, cv2.LINE_AA)
    # extra info
    if extra:
        cv2.putText(frame, extra, (w // 2 - 100, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, primary, 2, cv2.LINE_AA)
    # bottom bar
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h - 38), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov2, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame,
                "FIST 1.2s = next mode  |  PALM = pause  |  Q = quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (110, 110, 110), 1, cv2.LINE_AA)
    # corner brackets
    br = 24; t = 2
    for px, py, dx, dy in [
        (0, 58, 1, 1), (w, 58, -1, 1),
        (0, h - 38, 1, -1), (w, h - 38, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * br, py), primary, t)
        cv2.line(frame, (px, py), (px, py + dy * br), primary, t)


def draw_mode_overlay(frame, mode, progress, w, h, primary):
    ov = frame.copy()
    cv2.rectangle(ov, (w // 2 - 180, h // 2 - 55),
                  (w // 2 + 180, h // 2 + 55), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame,
                  (w // 2 - 180, h // 2 - 55),
                  (w // 2 + 180, h // 2 + 55), primary, 2)
    cv2.putText(frame, "MODE", (w // 2 - 155, h // 2 - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, mode, (w // 2 - 155, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, primary, 2, cv2.LINE_AA)
    bw = int(360 * progress)
    cv2.rectangle(frame,
                  (w // 2 - 180, h // 2 + 40),
                  (w // 2 - 180 + bw, h // 2 + 50), primary, -1)


def draw_volume_bar(frame, vol_pct, w, h, color):
    bh = int((h - 100) * vol_pct)
    bx = w - 35
    cv2.rectangle(frame, (bx, 58), (bx + 22, h - 40), (30, 30, 30), -1)
    cv2.rectangle(frame, (bx, h - 40 - bh), (bx + 22, h - 40), color, -1)
    cv2.rectangle(frame, (bx, 58), (bx + 22, h - 40), color, 1)
    cv2.putText(frame, f"{int(vol_pct * 100)}%", (bx - 10, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


#  MEDIAPIPE SETUP

latest_result = None


def result_callback(result, _, ts):
    global latest_result
    latest_result = result


if not os.path.exists(MODEL_FILE):
    print("Downloading hand landmarker model (~8 MB)…")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_FILE,
    )
    print("Download complete.\n")

