

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

