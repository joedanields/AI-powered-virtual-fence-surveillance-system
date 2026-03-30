# AI Virtual Fence Intrusion Detection System

A production-quality, real-time AI surveillance system that detects when humans cross user-defined boundaries in a camera feed and triggers multi-channel alerts.

---

## System Architecture

```
Camera Input (Webcam / RTSP / Video File)
        │
        ▼
  Frame Capture (OpenCV, threaded)
        │
        ▼
  Human Detection (YOLOv8 — person class only)
        │
        ▼
  Boundary Check (cv2.pointPolygonTest)
        │
        ▼
  Intrusion Engine (anti-spam cooldown)
        │
        ├──▶ Console Alert
        ├──▶ Alarm Sound
        ├──▶ Screenshot Evidence
        ├──▶ SQLite Event Log
        ├──▶ Telegram Notification
        ├──▶ Email SMTP Alert
        │
        ▼
  Flask Web Dashboard (event history, stats)
```

---

## Project Structure

```
├── main.py                 # Entry point — surveillance pipeline
├── config.py               # Centralized configuration (dataclasses)
├── camera.py               # Threaded video capture with auto-reconnect
├── detector.py             # YOLOv8 human detector wrapper
├── boundary.py             # Virtual fence zone manager + interactive drawer
├── intrusion_engine.py     # Core decision engine (detection + boundary + cooldown)
├── alert_system.py         # Multi-channel alert dispatcher
├── database.py             # SQLite intrusion event logger
├── visualization.py        # OpenCV overlay rendering (boxes, HUD, warnings)
├── web/
│   ├── __init__.py
│   ├── app.py              # Flask dashboard application
│   └── templates/
│       ├── base.html       # Base template with dark-themed UI
│       ├── index.html      # Dashboard home (stats + recent events)
│       ├── events.html     # Paginated event list
│       └── event_detail.html  # Single event view with screenshot
├── requirements.txt
├── assets/                 # Alarm sound file (place alarm.wav here)
├── evidence/               # Auto-created — saved screenshots & clips
├── data/                   # Auto-created — SQLite database
├── boundaries/             # Auto-created — saved zone polygons (JSON)
└── logs/                   # Auto-created — application logs
```

---

## Quick Start

### 1. Prerequisites

- Python 3.9+
- pip

### 2. Install Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd AI-powered-virtual-fence-surveillance-system

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install packages
pip install -r requirements.txt
```

> **Note:** On first run, the YOLOv8 nano model (`yolov8n.pt`) will be downloaded automatically (~6 MB).

### 3. Run the System

**Webcam (default):**
```bash
python main.py
```

**RTSP camera:**
```bash
python main.py --source "rtsp://user:pass@192.168.1.100:554/stream"
```

**Video file:**
```bash
python main.py --source "path/to/video.mp4"
```

**With web dashboard:**
```bash
python main.py --dashboard
# Dashboard at http://localhost:5000
```

**Full options:**
```bash
python main.py \
  --source 0 \
  --model yolov8n.pt \
  --confidence 0.45 \
  --camera-id cam_lobby \
  --dashboard \
  --telegram-token "BOT_TOKEN" \
  --telegram-chat "CHAT_ID"
```

### 4. Draw the Virtual Fence

On first run (or if no saved zones exist), an interactive drawing window opens:
- **Left-click** — add a polygon vertex
- **Right-click** — undo the last vertex
- **C** — confirm the polygon (minimum 3 points)
- **R** — reset all points
- **Q / ESC** — cancel

During live monitoring:
- **D** — draw a new zone
- **R** — reset all zones
- **Q / ESC** — quit

---

## Module Descriptions

### `config.py`
Centralized configuration using Python dataclasses. All tunable parameters (camera settings, model paths, alert credentials, display options) are defined here. Environment variables override defaults for production.

### `camera.py`
Thread-safe video capture with automatic reconnection. Runs frame grabbing in a daemon thread to prevent I/O blocking. Supports webcam indices, RTSP URLs, and local video files. Provides real-time FPS measurement.

### `detector.py`
Wraps the Ultralytics YOLOv8 model. Lazy-loads the model on first inference. Filters detections to person-class only (COCO class 0). Returns structured `Detection` objects with bounding box, confidence, center/bottom-center point helpers.

### `boundary.py`
Manages multiple named polygon zones with JSON persistence. Provides `cv2.pointPolygonTest`-based containment checks. Includes `BoundaryDrawer` — an interactive OpenCV mouse-callback tool for drawing polygons on a video frame.

### `intrusion_engine.py`
Core decision engine. Maps each detection to its reference point, tests against all zones, applies per-zone cooldown (default 10 seconds) to suppress duplicate alerts, dispatches alerts, and logs events. Supports time-based monitoring schedules.

### `alert_system.py`
Multi-channel alert dispatcher. Console logging, audible alarm (WAV playback), JPEG screenshot capture, optional MP4 video clip recording, Telegram Bot API, and SMTP email. Network-bound alerts run asynchronously in daemon threads.

### `database.py`
Thread-safe SQLite database using WAL journal mode. Stores intrusion events with fields: event_id, timestamp, camera_id, zone_name, screenshot_path, video_path, num_persons, status, notes. Supports CRUD operations and daily statistics.

### `visualization.py`
OpenCV overlay rendering — bounding boxes, person labels, reference points, zone polygons with semi-transparent fill, flashing intrusion warning banner, and a HUD showing FPS, inference time, detection count, and monitoring status.

### `web/app.py`
Flask web dashboard with a dark-themed UI. Routes: dashboard home (stats + recent events), paginated event list, event detail with screenshot, JSON API endpoints for events and statistics, evidence file serving.

### `main.py`
Entry point that wires all modules together into a `SurveillancePipeline`. Supports CLI arguments for source, model, confidence, camera ID, Telegram credentials. Handles graceful shutdown on SIGINT/SIGTERM.

---

## Configuration

Edit `config.py` or use CLI arguments / environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera.source` | `"0"` | Webcam index, RTSP URL, or video path |
| `detector.model_path` | `"yolov8n.pt"` | YOLOv8 model (n/s/m/l/x) |
| `detector.confidence_threshold` | `0.45` | Min detection confidence |
| `intrusion.cooldown_seconds` | `10.0` | Anti-spam cooldown per zone |
| `alert.telegram_enabled` | `False` | Enable Telegram notifications |
| `alert.email_enabled` | `False` | Enable SMTP email alerts |
| `schedule.enabled` | `False` | Time-based monitoring |
| `schedule.active_start` | `"22:00"` | Monitoring start (night mode) |
| `schedule.active_end` | `"06:00"` | Monitoring end |

### Telegram Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) and get the token.
2. Get your chat ID by messaging [@userinfobot](https://t.me/userinfobot).
3. Set environment variables or pass via CLI:
   ```bash
   export TELEGRAM_BOT_TOKEN="your_token"
   export TELEGRAM_CHAT_ID="your_chat_id"
   ```

### Email Setup

Set environment variables:
```bash
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="you@gmail.com"
export SMTP_PASSWORD="app_password"
export EMAIL_RECIPIENT="alerts@example.com"
```

---

## Performance Notes

- **CPU:** YOLOv8n runs at 20–30+ FPS on modern CPUs.
- **GPU:** Install `torch` with CUDA support and set `detector.device = "cuda"` for 60+ FPS.
- **Resolution:** Lower `camera.resolution` or `detector.img_size` for faster processing.
- **Model size:** `yolov8n.pt` (fastest) → `yolov8s.pt` → `yolov8m.pt` → `yolov8l.pt` (most accurate).

---

## License

MIT
