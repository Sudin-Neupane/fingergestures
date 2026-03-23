

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

