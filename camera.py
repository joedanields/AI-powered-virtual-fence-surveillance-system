"""
Camera module — thread-safe video capture with auto-reconnect.

Supports webcam indices, RTSP URLs, and local video files.
Runs frame grabbing in a dedicated daemon thread to prevent
I/O blocking in the main processing loop.
"""

import cv2
import time
import logging
import threading
import numpy as np
from typing import Optional, Tuple

from config import CameraConfig

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Asynchronous video stream reader with automatic reconnection.

    Usage:
        cam = CameraStream(config)
        cam.start()
        ok, frame = cam.read()
        cam.stop()
    """

    def __init__(self, config: CameraConfig):
        self._config = config
        self._source = self._parse_source(config.source)
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._grabbed: bool = False
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fps: float = 0.0
        self._frame_count: int = 0

    # ── public interface ───────────────────────
    def start(self) -> "CameraStream":
        """Open the video source and begin background frame grabbing."""
        self._open()
        self._running.set()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        logger.info("CameraStream started [%s] source=%s", self._config.camera_id, self._source)
        return self

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the most recent frame (thread-safe)."""
        with self._lock:
            if self._frame is None:
                return False, None
            return self._grabbed, self._frame.copy()

    def stop(self):
        """Signal the reader thread to exit and release the capture."""
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._release()
        logger.info("CameraStream stopped [%s]", self._config.camera_id)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return self._config.resolution

    # ── private helpers ────────────────────────
    @staticmethod
    def _parse_source(source: str):
        """Convert '0' / '1' to int for webcam; leave strings as-is."""
        try:
            return int(source)
        except ValueError:
            return source

    def _open(self):
        """Initialise cv2.VideoCapture with optimal settings."""
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise ConnectionError(f"Cannot open video source: {self._source}")

        # Apply settings
        w, h = self._config.resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._config.buffer_size)
        logger.info("VideoCapture opened → %s  (%dx%d @ %d fps)",
                     self._source, w, h, self._config.fps)

    def _release(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def _reconnect(self):
        """Attempt to reconnect with exponential back-off."""
        self._release()
        for attempt in range(1, self._config.max_reconnect_attempts + 1):
            delay = min(self._config.reconnect_delay * attempt, 30)
            logger.warning("Reconnecting [%s] attempt %d/%d in %.1fs …",
                           self._config.camera_id, attempt,
                           self._config.max_reconnect_attempts, delay)
            time.sleep(delay)
            try:
                self._open()
                logger.info("Reconnected [%s]", self._config.camera_id)
                return True
            except ConnectionError:
                continue
        logger.error("Failed to reconnect [%s] after %d attempts",
                     self._config.camera_id, self._config.max_reconnect_attempts)
        return False

    def _update_loop(self):
        """Background thread: continuously grab the latest frame."""
        fps_timer = time.time()
        fps_counter = 0

        while self._running.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._reconnect():
                    self._running.clear()
                    break
                continue

            grabbed, frame = self._cap.read()
            if not grabbed:
                # End of file or stream drop
                if isinstance(self._source, str) and not self._source.startswith("rtsp"):
                    # Video file — loop or stop
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if not self._reconnect():
                    self._running.clear()
                    break
                continue

            with self._lock:
                self._grabbed = grabbed
                self._frame = frame

            # FPS calculation
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self._fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()

            self._frame_count += 1
