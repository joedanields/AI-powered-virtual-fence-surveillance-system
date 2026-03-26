"""
Alert system module — multi-channel notification dispatcher.

Supports:
  • Console logging
  • Audible alarm (wav playback)
  • Screenshot evidence capture
  • Optional short video clip
  • Telegram Bot API
  • SMTP email
"""

import os
import logging
import smtplib
import threading
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import AlertConfig

logger = logging.getLogger(__name__)


class AlertDispatcher:
    """
    Dispatches intrusion alerts across configured channels.

    All network-bound alerts (Telegram, email) are sent asynchronously
    to avoid blocking the real-time pipeline.
    """

    def __init__(self, config: AlertConfig):
        self._config = config
        self._evidence_dir = Path(config.evidence_path)
        self._evidence_dir.mkdir(parents=True, exist_ok=True)

    # ── public dispatch ────────────────────────
    def trigger(
        self,
        frame: np.ndarray,
        camera_id: str,
        zone_name: str = "default",
        num_persons: int = 1,
        timestamp: Optional[datetime] = None,
    ) -> dict:
        """
        Fire all enabled alert channels.

        Returns a dict with paths / status for each channel.
        """
        ts = timestamp or datetime.now()
        ts_str = ts.strftime("%Y%m%d_%H%M%S")
        message = (
            f"⚠ Intrusion Detected — "
            f"Human crossed boundary [{zone_name}] at {ts.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(camera: {camera_id}, persons: {num_persons})"
        )

        result = {
            "timestamp": ts.isoformat(),
            "camera_id": camera_id,
            "zone_name": zone_name,
            "screenshot_path": "",
            "video_path": "",
        }

        # 1. Console
        if self._config.console_enabled:
            self._alert_console(message)

        # 2. Sound
        if self._config.sound_enabled:
            self._alert_sound()

        # 3. Screenshot
        screenshot_path = ""
        if self._config.screenshot_enabled:
            screenshot_path = self._save_screenshot(frame, camera_id, ts_str)
            result["screenshot_path"] = screenshot_path

        # 4. Telegram (async)
        if self._config.telegram_enabled and self._config.telegram_bot_token:
            threading.Thread(
                target=self._send_telegram,
                args=(message, screenshot_path),
                daemon=True,
            ).start()

        # 5. Email (async)
        if self._config.email_enabled and self._config.smtp_username:
            threading.Thread(
                target=self._send_email,
                args=(message, screenshot_path),
                daemon=True,
            ).start()

        return result

    # ── individual channels ────────────────────
    @staticmethod
    def _alert_console(message: str):
        logger.warning(message)
        print(f"\n{'='*60}\n{message}\n{'='*60}\n")

    def _alert_sound(self):
        """Play alarm wav in a background thread (non-blocking)."""
        try:
            sound_path = self._config.alarm_sound_path
            if not os.path.isfile(sound_path):
                logger.debug("Alarm sound file not found: %s — using beep fallback", sound_path)
                self._beep_fallback()
                return
            # Cross-platform wav playback
            threading.Thread(target=self._play_wav, args=(sound_path,), daemon=True).start()
        except Exception as exc:
            logger.error("Sound alert failed: %s", exc)

    @staticmethod
    def _play_wav(path: str):
        """Attempt to play a .wav file using available system methods."""
        try:
            import platform
            if platform.system() == "Windows":
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                os.system(f'aplay "{path}" 2>/dev/null || afplay "{path}" 2>/dev/null &')
        except Exception as exc:
            logger.error("WAV playback error: %s", exc)

    @staticmethod
    def _beep_fallback():
        """Simple console beep."""
        try:
            import platform
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 500)
            else:
                print("\a", end="", flush=True)
        except Exception:
            print("\a", end="", flush=True)

    def _save_screenshot(self, frame: np.ndarray, camera_id: str, ts_str: str) -> str:
        """Save intrusion frame as JPEG evidence."""
        filename = f"intrusion_{camera_id}_{ts_str}.jpg"
        filepath = str(self._evidence_dir / filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.info("Screenshot saved → %s", filepath)
        return filepath

    # ── video clip capture (called externally) ─
    def start_video_clip(self, camera_id: str, fps: float = 20.0,
                         resolution: tuple = (1280, 720)) -> Optional["VideoClipWriter"]:
        """Begin recording a short evidence clip. Returns a writer handle."""
        if not self._config.video_clip_enabled:
            return None
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"clip_{camera_id}_{ts_str}.mp4"
        filepath = str(self._evidence_dir / filename)
        return VideoClipWriter(filepath, fps, resolution, self._config.video_clip_duration)

    # ── Telegram ───────────────────────────────
    def _send_telegram(self, message: str, image_path: str = ""):
        """Send Telegram text and optional photo."""
        try:
            import requests
        except ImportError:
            logger.error("requests library not installed — Telegram alerts disabled")
            return

        token = self._config.telegram_bot_token
        chat_id = self._config.telegram_chat_id
        base_url = f"https://api.telegram.org/bot{token}"

        try:
            # Text message
            requests.post(f"{base_url}/sendMessage", data={
                "chat_id": chat_id, "text": message,
            }, timeout=10)

            # Photo
            if image_path and os.path.isfile(image_path):
                with open(image_path, "rb") as photo:
                    requests.post(f"{base_url}/sendPhoto", data={
                        "chat_id": chat_id,
                    }, files={"photo": photo}, timeout=30)

            logger.info("Telegram alert sent")
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)

    # ── Email ──────────────────────────────────
    def _send_email(self, message: str, image_path: str = ""):
        """Send alert email via SMTP."""
        try:
            msg = MIMEMultipart()
            msg["From"] = self._config.smtp_username
            msg["To"] = self._config.email_recipient
            msg["Subject"] = "⚠ Virtual Fence Intrusion Alert"
            msg.attach(MIMEText(message, "plain"))

            if image_path and os.path.isfile(image_path):
                with open(image_path, "rb") as f:
                    img = MIMEImage(f.read(), name=os.path.basename(image_path))
                    msg.attach(img)

            with smtplib.SMTP(self._config.smtp_server, self._config.smtp_port) as server:
                server.starttls()
                server.login(self._config.smtp_username, self._config.smtp_password)
                server.send_message(msg)
            logger.info("Email alert sent to %s", self._config.email_recipient)
        except Exception as exc:
            logger.error("Email send failed: %s", exc)


class VideoClipWriter:
    """
    Accumulates frames into an MP4 clip for a fixed duration.

    Usage:
        writer = VideoClipWriter(path, fps, resolution, duration)
        writer.write(frame)   # call every frame
        if writer.is_done:
            writer.release()
    """

    def __init__(self, filepath: str, fps: float, resolution: tuple, duration_sec: int):
        self._filepath = filepath
        self._max_frames = int(fps * duration_sec)
        self._count = 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(filepath, fourcc, fps, resolution)
        logger.info("Video clip recording started → %s", filepath)

    def write(self, frame: np.ndarray):
        if self._count < self._max_frames:
            self._writer.write(frame)
            self._count += 1

    @property
    def is_done(self) -> bool:
        return self._count >= self._max_frames

    @property
    def filepath(self) -> str:
        return self._filepath

    def release(self):
        self._writer.release()
        logger.info("Video clip saved → %s (%d frames)", self._filepath, self._count)
