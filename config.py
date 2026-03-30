"""
Configuration module for the AI Virtual Fence Surveillance System.

Centralizes all tunable parameters, paths, and credentials.
Environment variables override defaults for production deployments.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
#  BASE PATHS
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "evidence"
DATABASE_DIR = BASE_DIR / "data"
BOUNDARY_DIR = BASE_DIR / "boundaries"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "web" / "static"
TEMPLATES_DIR = BASE_DIR / "web" / "templates"

# Ensure directories exist at import time
for _dir in [EVIDENCE_DIR, DATABASE_DIR, BOUNDARY_DIR, LOGS_DIR, MODELS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
#  CAMERA CONFIGURATION
# ──────────────────────────────────────────────
@dataclass
class CameraConfig:
    """Configuration for a single camera source."""
    camera_id: str = "cam_01"
    source: str = "0"                          # 0 = webcam, rtsp://..., or file path
    resolution: tuple = (1280, 720)
    fps: int = 30
    reconnect_delay: float = 5.0               # seconds before reconnect attempt
    max_reconnect_attempts: int = 10
    buffer_size: int = 1                       # OpenCV buffer size (1 = minimal latency)


# ──────────────────────────────────────────────
#  DETECTOR CONFIGURATION
# ──────────────────────────────────────────────
@dataclass
class DetectorConfig:
    """YOLOv8 detection parameters."""
    model_path: str = "yolov8n.pt"             # nano model for CPU; swap to yolov8s/m/l for GPU
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.50
    target_classes: List[int] = field(default_factory=lambda: [0])  # COCO class 0 = person
    device: str = ""                           # "" = auto (CUDA if available, else CPU)
    img_size: int = 640
    half_precision: bool = False               # FP16 — enable only on CUDA
    max_detections: int = 50


# ──────────────────────────────────────────────
#  BOUNDARY / VIRTUAL FENCE
# ──────────────────────────────────────────────
@dataclass
class BoundaryConfig:
    """Virtual fence drawing and persistence."""
    min_points: int = 3
    line_color: tuple = (0, 0, 255)            # BGR red
    line_thickness: int = 2
    fill_alpha: float = 0.25
    point_radius: int = 6
    point_color: tuple = (0, 255, 0)           # BGR green
    save_path: str = str(BOUNDARY_DIR / "zones.json")


# ──────────────────────────────────────────────
#  INTRUSION DETECTION
# ──────────────────────────────────────────────
@dataclass
class IntrusionConfig:
    """Anti-spam and detection behaviour."""
    cooldown_seconds: float = 10.0             # min gap between repeated alerts per person
    overlap_iou_threshold: float = 0.5         # for tracking same bounding box across frames
    center_point_method: str = "bottom_center" # "center" or "bottom_center" (feet position)


# ──────────────────────────────────────────────
#  ALERT SYSTEM
# ──────────────────────────────────────────────
@dataclass
class AlertConfig:
    """Alert channels and their parameters."""
    # Console
    console_enabled: bool = True

    # Sound
    sound_enabled: bool = True
    alarm_sound_path: str = str(BASE_DIR / "assets" / "alarm.wav")

    # Screenshot evidence
    screenshot_enabled: bool = True
    evidence_path: str = str(EVIDENCE_DIR)
    video_clip_enabled: bool = False
    video_clip_duration: int = 10               # seconds

    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Email / SMTP
    email_enabled: bool = False
    smtp_server: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    email_recipient: str = os.getenv("EMAIL_RECIPIENT", "")


# ──────────────────────────────────────────────
#  DATABASE
# ──────────────────────────────────────────────
@dataclass
class DatabaseConfig:
    """SQLite event log database."""
    db_path: str = str(DATABASE_DIR / "intrusions.db")


# ──────────────────────────────────────────────
#  WEB DASHBOARD
# ──────────────────────────────────────────────
@dataclass
class DashboardConfig:
    """Flask web dashboard."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    secret_key: str = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
    events_per_page: int = 20


# ──────────────────────────────────────────────
#  DISPLAY / VISUALIZATION
# ──────────────────────────────────────────────
@dataclass
class DisplayConfig:
    """Overlay rendering parameters."""
    bbox_color: tuple = (0, 255, 0)
    bbox_thickness: int = 2
    label_font_scale: float = 0.6
    label_color: tuple = (255, 255, 255)
    warning_color: tuple = (0, 0, 255)
    warning_font_scale: float = 1.2
    show_fps: bool = True
    window_name: str = "AI Virtual Fence"


# ──────────────────────────────────────────────
#  TIME-BASED SECURITY MODE
# ──────────────────────────────────────────────
@dataclass
class ScheduleConfig:
    """Night / time-based monitoring schedule."""
    enabled: bool = False
    active_start: str = "22:00"                # HH:MM (24-hour)
    active_end: str = "06:00"


# ──────────────────────────────────────────────
#  AGGREGATE CONFIG
# ──────────────────────────────────────────────
@dataclass
class AppConfig:
    """Top-level application configuration container."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    intrusion: IntrusionConfig = field(default_factory=IntrusionConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)


def load_config() -> AppConfig:
    """
    Factory that returns the global AppConfig.
    Extend this to load from YAML / JSON / .env files in production.
    """
    return AppConfig()
