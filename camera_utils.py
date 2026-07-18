"""
Shared camera-selection utilities for all camera demos in this repo.

Every camera demo accepts --video-source (or the VIDEO_SOURCE environment
variable). When neither is given, resolve_video_source() scans the available
cameras and prompts the user to pick one — with a snapshot preview window when
a GUI is available, so it's easy to tell the built-in webcam apart from an
iPhone Continuity Camera.
"""

import json
import os
import subprocess
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

CameraInfo = Tuple[int, int, int, Optional[np.ndarray]]


def open_capture(source) -> cv2.VideoCapture:
    """Create a cv2.VideoCapture using either a numeric index or URL"""
    source = str(source)
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def list_available_cameras(max_index: int = 5) -> List[CameraInfo]:
    """Probe camera indices and return (index, width, height, preview_frame) per camera"""
    cameras = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ret, frame = cap.read()
            cameras.append((index, width, height, frame if ret else None))
        cap.release()
    return cameras


def get_macos_camera_names() -> List[str]:
    """Best-effort camera names from system_profiler (macOS only).

    The order does not necessarily match OpenCV's index order, so treat these
    as a hint about what hardware is present, not as index labels.
    """
    if sys.platform != "darwin":
        return []
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=15, check=True
        )
        cameras = json.loads(result.stdout).get("SPCameraDataType", [])
        return [cam.get("_name", "Unknown camera") for cam in cameras]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []


def build_camera_preview(cameras: List[CameraInfo], thumb_width: int = 320) -> np.ndarray:
    """Build a side-by-side strip of one snapshot per camera, labeled by index"""
    thumbs = []
    for index, _, _, frame in cameras:
        if frame is None:
            frame = np.zeros((240, thumb_width, 3), dtype=np.uint8)
        scale = thumb_width / frame.shape[1]
        thumb = cv2.resize(frame, (thumb_width, max(1, int(frame.shape[0] * scale))))
        cv2.putText(thumb, f"[{index}]", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        thumbs.append(thumb)
    max_height = max(t.shape[0] for t in thumbs)
    padded = [
        cv2.copyMakeBorder(t, 0, max_height - t.shape[0], 0, 2, cv2.BORDER_CONSTANT)
        for t in thumbs
    ]
    return np.hstack(padded)


def _gui_available() -> bool:
    """Return True if OpenCV can create GUI windows"""
    if sys.platform != "darwin" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    test_window = "_camera_utils_gui_test"
    try:
        cv2.namedWindow(test_window)
        cv2.destroyWindow(test_window)
        return True
    except cv2.error:
        return False


def _prompt_via_window(cameras: List[CameraInfo]) -> Optional[str]:
    """Show snapshot previews and let the user pick a camera by pressing its number key"""
    valid = {str(index) for index, _, _, _ in cameras}
    default = str(cameras[0][0])
    window = "Select camera"
    print("Preview window opened. Press the number key of the camera to use "
          f"('q' or Esc keeps camera {default}).")
    try:
        cv2.imshow(window, build_camera_preview(cameras))
        choice = default
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord('q'), 27):
                break
            if key != 255 and chr(key) in valid:
                choice = chr(key)
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except cv2.error:
        return None
    try:
        cv2.destroyWindow(window)
        cv2.waitKey(1)
    except cv2.error:
        pass
    return choice


def _prompt_via_terminal(cameras: List[CameraInfo]) -> Optional[str]:
    """Ask for a camera index on stdin; None when stdin is not interactive"""
    if sys.stdin is None or not sys.stdin.isatty():
        return None
    valid = {str(index) for index, _, _, _ in cameras}
    default = str(cameras[0][0])
    while True:
        choice = input(f"Select a camera index [{default}]: ").strip()
        if choice == "":
            return default
        if choice in valid:
            return choice
        print(f"Invalid choice '{choice}'. Options: {', '.join(sorted(valid))}")


def prompt_for_camera() -> Optional[str]:
    """Scan for cameras and let the user pick one; None when no camera is found"""
    print("No video source given. Scanning for cameras...")
    cameras = list_available_cameras()
    if not cameras:
        print("No cameras found.")
        return None

    names = get_macos_camera_names()
    if names:
        print("Detected camera hardware (order may not match indices below): "
              + ", ".join(names))
    print("Available cameras:")
    for index, width, height, _ in cameras:
        print(f"  [{index}] {width}x{height}")

    if len(cameras) == 1:
        print(f"Only one camera found. Using camera {cameras[0][0]}.")
        return str(cameras[0][0])

    choice = None
    if _gui_available():
        choice = _prompt_via_window(cameras)
    if choice is None:
        choice = _prompt_via_terminal(cameras)
    if choice is None:
        choice = str(cameras[0][0])
        print(f"No selection possible. Using camera {choice}.")
    else:
        print(f"Using camera {choice}.")
    return choice


def resolve_video_source(cli_value: Optional[str]) -> str:
    """Resolve the video source: CLI flag > VIDEO_SOURCE env var > interactive prompt > "0" """
    if cli_value:
        return str(cli_value)
    env_value = os.getenv("VIDEO_SOURCE")
    if env_value:
        return env_value
    selected = prompt_for_camera()
    if selected is not None:
        return selected
    return "0"


VIDEO_SOURCE_HELP = (
    "Camera index (e.g. 0) or stream URL. If omitted, available cameras are "
    "listed and you are prompted to pick one."
)
