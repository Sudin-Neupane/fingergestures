

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