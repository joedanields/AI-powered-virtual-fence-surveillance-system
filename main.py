"""
Main pipeline — orchestrates the full surveillance system.

Wires together Camera → Detector → Boundary → Intrusion Engine → Display.
Supports both headless mode (for server deployment) and GUI mode (with OpenCV window).
"""

import argparse
import logging
import sys
import time
import signal
from datetime import datetime

import cv2
import numpy as np

from config import load_config, AppConfig
from camera import CameraStream
from detector import HumanDetector
from boundary import BoundaryManager, BoundaryDrawer
from alert_system import AlertDispatcher
from database import IntrusionDatabase
from intrusion_engine import IntrusionEngine
from visualization import draw_detections, draw_intrusion_warning, draw_hud

# ──────────────────────────────────────────────
#  LOGGING SETUP
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/surveillance.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


class SurveillancePipeline:
    """
    End-to-end surveillance pipeline.

    Lifecycle:
        pipeline = SurveillancePipeline(config)
        pipeline.setup()
        pipeline.run()      # blocks until quit
        pipeline.teardown()
    """

    def __init__(self, config: AppConfig):
        self.cfg = config
        self.camera: CameraStream = None
        self.detector: HumanDetector = None
        self.boundary_mgr: BoundaryManager = None
        self.alert_dispatcher: AlertDispatcher = None
        self.database: IntrusionDatabase = None
        self.engine: IntrusionEngine = None
        self._running = False

        # ── Edit-mode state ──────────────────
        self._edit_mode = False
        self._selected_zone: str = None       # currently selected zone name
        self._dragging_vertex: int = None     # vertex index being dragged
        self._show_help = False               # toggle help overlay

    def setup(self):
        """Initialize all subsystems."""
        logger.info("=" * 60)
        logger.info("  AI Virtual Fence Intrusion Detection System")
        logger.info("=" * 60)

        # Database
        self.database = IntrusionDatabase(self.cfg.database)

        # Camera
        self.camera = CameraStream(self.cfg.camera)
        self.camera.start()

        # Wait for first frame
        logger.info("Waiting for first frame...")
        for _ in range(50):
            ok, frame = self.camera.read()
            if ok:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Could not read initial frame from camera")

        # Boundary manager
        self.boundary_mgr = BoundaryManager(self.cfg.boundary)

        # If no zones configured, launch interactive drawing
        if not self.boundary_mgr.has_zones():
            logger.info("No boundary zones found — launching drawing tool")
            self._draw_initial_boundary(frame)

        # Detector (loads model on first inference)
        self.detector = HumanDetector(self.cfg.detector)

        # Alert system
        self.alert_dispatcher = AlertDispatcher(self.cfg.alert)

        # Intrusion engine
        self.engine = IntrusionEngine(
            config=self.cfg.intrusion,
            schedule_config=self.cfg.schedule,
            boundary_mgr=self.boundary_mgr,
            alert_dispatcher=self.alert_dispatcher,
            database=self.database,
            camera_id=self.cfg.camera.camera_id,
        )

        logger.info("All subsystems initialized ✓")

    def _draw_initial_boundary(self, frame):
        """Launch the interactive boundary drawing tool."""
        drawer = BoundaryDrawer(frame, self.cfg.boundary, zone_name="zone_1")
        points = drawer.run()
        if points:
            self.boundary_mgr.add_zone("zone_1", points)
            logger.info("Boundary zone_1 created with %d points", len(points))
        else:
            logger.warning("No boundary drawn — the system will run without intrusion detection")

    def run(self, headless: bool = False):
        """
        Main processing loop.

        Args:
            headless: If True, skip OpenCV GUI window (for server / Docker).
        """
        self._running = True
        logger.info("Pipeline running — press 'h' for controls help")

        fps_timer = time.time()
        fps_counter = 0
        display_fps = 0.0

        # Register mouse callback for edit-mode vertex dragging
        if not headless:
            cv2.namedWindow(self.cfg.display.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.cfg.display.window_name, self._mouse_callback)

        while self._running:
            ok, frame = self.camera.read()
            if not ok:
                time.sleep(0.01)
                continue

            # ── Detection ──────────────────────
            detections = self.detector.detect(frame)

            # ── Intrusion evaluation (skip in edit mode) ──
            if self._edit_mode:
                intrusions = []
            else:
                intrusions = self.engine.process(frame, detections)

            # ── Visualization ──────────────────
            if not headless:
                vis_frame = frame.copy()
                vis_frame = self.boundary_mgr.draw_zones(
                    vis_frame,
                    selected_zone=self._selected_zone,
                    edit_mode=self._edit_mode,
                )
                vis_frame = draw_detections(vis_frame, detections, self.cfg.display)
                vis_frame = draw_intrusion_warning(vis_frame, intrusions, self.cfg.display)

                # HUD
                fps_counter += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    display_fps = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now

                vis_frame = draw_hud(
                    vis_frame,
                    fps=display_fps,
                    inference_ms=self.detector.inference_time_ms,
                    num_detections=len(detections),
                    num_zones=len(self.boundary_mgr.get_all_zones()),
                    monitoring_active=self.engine._is_monitoring_active(),
                    config=self.cfg.display,
                )

                # Mode indicator bar
                vis_frame = self._draw_mode_bar(vis_frame)

                # Help overlay
                if self._show_help:
                    vis_frame = self._draw_help_overlay(vis_frame)

                cv2.imshow(self.cfg.display.window_name, vis_frame)

                # Keyboard handling
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key, frame)
                if key == ord("q") or key == 27:
                    break

        self.teardown()

    # ── Mouse callback for vertex dragging ─────
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for live vertex editing."""
        if not self._edit_mode or not self._selected_zone:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drag if near a vertex
            idx = self.boundary_mgr.find_nearest_vertex(
                self._selected_zone, (x, y), max_dist=25.0
            )
            if idx is not None:
                self._dragging_vertex = idx

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._dragging_vertex is not None:
                self.boundary_mgr.move_vertex(
                    self._selected_zone, self._dragging_vertex, (x, y)
                )

        elif event == cv2.EVENT_LBUTTONUP:
            if self._dragging_vertex is not None:
                self.boundary_mgr.move_vertex(
                    self._selected_zone, self._dragging_vertex, (x, y)
                )
                self._dragging_vertex = None
                logger.info("Vertex moved — zone '%s' updated", self._selected_zone)

    # ── Keyboard handler ───────────────────────
    def _handle_key(self, key, frame):
        """Process keyboard input for zone management."""
        if key == 255:  # no key
            return

        # ── Toggle help ──
        if key == ord("h"):
            self._show_help = not self._show_help
            return

        # ── Toggle edit mode ──
        if key == ord("e"):
            self._edit_mode = not self._edit_mode
            if self._edit_mode:
                # Auto-select first zone if none selected
                names = self.boundary_mgr.zone_names()
                if names and not self._selected_zone:
                    self._selected_zone = names[0]
                logger.info("EDIT MODE ON — selected: %s", self._selected_zone)
            else:
                self._dragging_vertex = None
                logger.info("EDIT MODE OFF — monitoring resumed")
            return

        # ── Draw new zone ──
        if key == ord("d"):
            self._draw_new_zone(frame)
            return

        # ── Reset all zones ──
        if key == ord("r") and not self._edit_mode:
            self._reset_zones()
            self._selected_zone = None
            return

        # ── Edit-mode specific keys ──
        if self._edit_mode:
            # Tab / 'n' = cycle through zones
            if key == ord("n") or key == 9:  # 9 = Tab
                self._cycle_selected_zone()
                return

            # 'x' = delete selected zone
            if key == ord("x"):
                self._delete_selected_zone()
                return

            # 'w' = redraw (replace) selected zone
            if key == ord("w"):
                self._redraw_selected_zone(frame)
                return

            # 'a' = add a vertex to selected zone (appended; user drags it after)
            if key == ord("a"):
                if self._selected_zone:
                    zone = self.boundary_mgr.get_zone(self._selected_zone)
                    if zone:
                        # Place new vertex at centroid
                        cx = int(np.mean([p[0] for p in zone.points]))
                        cy = int(np.mean([p[1] for p in zone.points]))
                        self.boundary_mgr.add_vertex(self._selected_zone, (cx, cy))
                        logger.info("Added vertex to '%s' — drag it to position", self._selected_zone)
                return

            # Backspace = remove last vertex from selected zone
            if key == 8:  # Backspace
                if self._selected_zone:
                    zone = self.boundary_mgr.get_zone(self._selected_zone)
                    if zone and len(zone.points) > self.cfg.boundary.min_points:
                        self.boundary_mgr.remove_vertex(
                            self._selected_zone, len(zone.points) - 1
                        )
                        logger.info("Removed last vertex from '%s'", self._selected_zone)
                    else:
                        logger.warning("Cannot remove — minimum %d vertices required",
                                       self.cfg.boundary.min_points)
                return

    # ── Zone management helpers ────────────────
    def _cycle_selected_zone(self):
        """Cycle to the next zone."""
        names = self.boundary_mgr.zone_names()
        if not names:
            self._selected_zone = None
            return
        if self._selected_zone in names:
            idx = (names.index(self._selected_zone) + 1) % len(names)
        else:
            idx = 0
        self._selected_zone = names[idx]
        logger.info("Selected zone: %s", self._selected_zone)

    def _delete_selected_zone(self):
        """Delete the currently selected zone."""
        if self._selected_zone:
            name = self._selected_zone
            self.boundary_mgr.remove_zone(name)
            logger.info("Deleted zone '%s'", name)
            # Select next zone or None
            names = self.boundary_mgr.zone_names()
            self._selected_zone = names[0] if names else None

    def _redraw_selected_zone(self, frame):
        """Redraw (replace) the currently selected zone."""
        if not self._selected_zone:
            return
        name = self._selected_zone
        points = self._open_drawer(frame, name)
        if points:
            self.boundary_mgr.add_zone(name, points)
            logger.info("Zone '%s' redrawn with %d points", name, len(points))
        else:
            logger.info("Redraw cancelled for '%s'", name)

    def _draw_new_zone(self, frame):
        """Interactive zone drawing in mid-stream."""
        idx = len(self.boundary_mgr.get_all_zones()) + 1
        name = f"zone_{idx}"
        points = self._open_drawer(frame, name)
        if points:
            self.boundary_mgr.add_zone(name, points)
            self._selected_zone = name
            logger.info("New zone '%s' added", name)

    def _open_drawer(self, frame, zone_name: str):
        """
        Open the BoundaryDrawer while safely disabling the main window
        to prevent mouse-callback conflicts on Windows.
        """
        win = self.cfg.display.window_name
        # Destroy the main window so its mouse callback doesn't steal clicks
        try:
            cv2.destroyWindow(win)
            cv2.waitKey(1)
        except Exception:
            pass

        drawer = BoundaryDrawer(frame, self.cfg.boundary, zone_name=zone_name)
        points = drawer.run()

        # Recreate the main window and re-attach mouse callback
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self._mouse_callback)
        return points

    def _reset_zones(self):
        """Remove all zones."""
        for name in list(self.boundary_mgr.zone_names()):
            self.boundary_mgr.remove_zone(name)
        self._selected_zone = None
        logger.info("All zones cleared")

    # ── On-screen overlays ─────────────────────
    def _draw_mode_bar(self, frame: np.ndarray) -> np.ndarray:
        """Draw a top status bar showing current mode and selected zone."""
        h, w = frame.shape[:2]
        bar_h = 32

        if self._edit_mode:
            # Orange bar for edit mode
            cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 140, 255), -1)
            zone_txt = self._selected_zone or "none"
            text = f"EDIT MODE | Zone: {zone_txt} | N=next  X=delete  W=redraw  A=add vertex  Bksp=rm vertex  E=exit edit"
            cv2.putText(frame, text, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # Dark semi-transparent bar for normal mode
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, bar_h), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            zones = len(self.boundary_mgr.get_all_zones())
            text = f"MONITORING | Zones: {zones} | H=help  D=new zone  E=edit  R=reset  Q=quit"
            cv2.putText(frame, text, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        return frame

    def _draw_help_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw a translucent help overlay listing all controls."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent dark background
        pad = 40
        cv2.rectangle(overlay, (pad, pad), (w - pad, h - pad), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        lines = [
            "=== KEYBOARD CONTROLS ===",
            "",
            "--- Normal Mode ---",
            "H        Toggle this help overlay",
            "D        Draw a new boundary zone",
            "E        Enter EDIT mode (drag vertices, manage zones)",
            "R        Reset ALL zones",
            "Q / ESC  Quit the application",
            "",
            "--- Edit Mode (press E first) ---",
            "N / Tab  Cycle through zones (select next)",
            "X        Delete the selected zone",
            "W        Redraw (replace) the selected zone",
            "A        Add a new vertex to the selected zone",
            "Bksp     Remove the last vertex from the selected zone",
            "E        Exit edit mode (resume monitoring)",
            "",
            "--- Mouse (Edit Mode) ---",
            "Click & drag a vertex handle to move it",
            "",
            "Press H to close",
        ]

        y = pad + 35
        for line in lines:
            color = (0, 200, 255) if line.startswith("===") or line.startswith("---") else (220, 220, 220)
            scale = 0.6 if line.startswith("===") else 0.5
            cv2.putText(frame, line, (pad + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += 24

        return frame

    def teardown(self):
        """Clean shutdown of all subsystems."""
        self._running = False
        if self.camera:
            self.camera.stop()
        cv2.destroyAllWindows()
        logger.info("Pipeline shut down cleanly")


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Virtual Fence Intrusion Detection System"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Camera source: 0 (webcam), rtsp://..., or path to video file",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="YOLOv8 model path (e.g. yolov8n.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--confidence", type=float, default=None,
        help="Detection confidence threshold (0-1)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without OpenCV GUI window",
    )
    parser.add_argument(
        "--camera-id", type=str, default=None,
        help="Camera identifier for multi-camera setups",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Start the Flask web dashboard in a background thread",
    )
    parser.add_argument(
        "--telegram-token", type=str, default=None,
        help="Telegram Bot API token",
    )
    parser.add_argument(
        "--telegram-chat", type=str, default=None,
        help="Telegram chat ID for alerts",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    # Override config from CLI args
    if args.source is not None:
        config.camera.source = args.source
    if args.model is not None:
        config.detector.model_path = args.model
    if args.confidence is not None:
        config.detector.confidence_threshold = args.confidence
    if args.camera_id is not None:
        config.camera.camera_id = args.camera_id
    if args.telegram_token:
        config.alert.telegram_enabled = True
        config.alert.telegram_bot_token = args.telegram_token
    if args.telegram_chat:
        config.alert.telegram_chat_id = args.telegram_chat

    # Start dashboard if requested
    if args.dashboard:
        import threading
        from web.app import create_app
        app = create_app(config)
        dash_thread = threading.Thread(
            target=lambda: app.run(
                host=config.dashboard.host,
                port=config.dashboard.port,
                debug=False,
                use_reloader=False,
            ),
            daemon=True,
        )
        dash_thread.start()
        logger.info("Web dashboard started at http://%s:%d",
                     config.dashboard.host, config.dashboard.port)

    # Build and run pipeline
    pipeline = SurveillancePipeline(config)

    # Graceful shutdown on SIGINT / SIGTERM
    def _signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        pipeline.teardown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        pipeline.setup()
        pipeline.run(headless=args.headless)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
    finally:
        pipeline.teardown()


if __name__ == "__main__":
    main()
