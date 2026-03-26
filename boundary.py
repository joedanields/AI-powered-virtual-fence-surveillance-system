"""
Boundary module — virtual fence zone management.

Supports multiple named polygon zones, persistence to JSON,
interactive mouse-based drawing via OpenCV, and live vertex editing.
"""

import json
import logging
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import BoundaryConfig

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """A single virtual fence polygon."""
    name: str
    points: List[Tuple[int, int]]   # ordered polygon vertices

    @property
    def polygon(self) -> np.ndarray:
        """Return points as a numpy array suitable for cv2 operations."""
        return np.array(self.points, dtype=np.int32)

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if a 2-D point lies inside (or on) the polygon."""
        result = cv2.pointPolygonTest(
            self.polygon.reshape((-1, 1, 2)),
            (float(point[0]), float(point[1])),
            measureDist=False,
        )
        return result >= 0  # >= 0 means inside or on border

    def is_valid(self, min_points: int = 3) -> bool:
        return len(self.points) >= min_points


class BoundaryManager:
    """
    Manages multiple virtual fence zones.

    Zones are persisted to a JSON file and can be reloaded across sessions.
    """

    def __init__(self, config: BoundaryConfig):
        self._config = config
        self._zones: Dict[str, Zone] = {}
        self._load()

    # ── zone CRUD ──────────────────────────────
    def add_zone(self, name: str, points: List[Tuple[int, int]]) -> bool:
        """Add or overwrite a zone. Returns True if the zone is valid."""
        zone = Zone(name=name, points=points)
        if not zone.is_valid(self._config.min_points):
            logger.warning("Zone '%s' rejected — needs ≥ %d points", name, self._config.min_points)
            return False
        self._zones[name] = zone
        self._save()
        logger.info("Zone '%s' saved with %d points", name, len(points))
        return True

    def remove_zone(self, name: str) -> bool:
        if name in self._zones:
            del self._zones[name]
            self._save()
            return True
        return False

    def get_zone(self, name: str) -> Optional[Zone]:
        return self._zones.get(name)

    def get_all_zones(self) -> Dict[str, Zone]:
        return dict(self._zones)

    def zone_names(self) -> List[str]:
        return list(self._zones.keys())

    def has_zones(self) -> bool:
        return bool(self._zones)

    # ── intrusion test ─────────────────────────
    def check_point(self, point: Tuple[int, int]) -> List[str]:
        """
        Return names of all zones that contain the given point.
        """
        violated = []
        for name, zone in self._zones.items():
            if zone.contains_point(point):
                violated.append(name)
        return violated

    # ── persistence ────────────────────────────
    def _save(self):
        data = {
            name: zone.points for name, zone in self._zones.items()
        }
        path = Path(self._config.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("Zones saved → %s", path)

    def _load(self):
        path = Path(self._config.save_path)
        if not path.exists():
            logger.info("No saved zones found at %s", path)
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for name, pts in data.items():
                zone = Zone(name=name, points=[tuple(p) for p in pts])
                if zone.is_valid(self._config.min_points):
                    self._zones[name] = zone
            logger.info("Loaded %d zone(s) from %s", len(self._zones), path)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to load zones: %s", exc)

    # ── drawing helpers ────────────────────────
    def draw_zones(self, frame: np.ndarray, selected_zone: Optional[str] = None,
                   edit_mode: bool = False) -> np.ndarray:
        """Render all zones on the frame (border + semi-transparent fill).
        
        Args:
            frame: The video frame.
            selected_zone: If set, highlight this zone differently.
            edit_mode: If True, draw draggable vertex handles.
        """
        overlay = frame.copy()
        for name, zone in self._zones.items():
            is_selected = (name == selected_zone)
            color = (0, 255, 255) if is_selected else self._config.line_color  # yellow if selected
            thickness = self._config.line_thickness + 2 if is_selected else self._config.line_thickness

            pts = zone.polygon.reshape((-1, 1, 2))
            # Fill
            fill_color = (0, 200, 200) if is_selected else self._config.line_color
            cv2.fillPoly(overlay, [zone.polygon], fill_color)
            # Border
            cv2.polylines(frame, [pts], isClosed=True,
                          color=color, thickness=thickness)
            # Label
            cx, cy = int(np.mean(zone.polygon[:, 0])), int(np.mean(zone.polygon[:, 1]))
            label = f"{name}" + (" [SELECTED]" if is_selected else "")
            cv2.putText(frame, label, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self._config.point_color, 2)

            # Draw vertex handles in edit mode
            if edit_mode and is_selected:
                for i, pt in enumerate(zone.points):
                    cv2.circle(frame, pt, self._config.point_radius + 2,
                               (255, 255, 0), -1)  # cyan handles
                    cv2.circle(frame, pt, self._config.point_radius + 2,
                               (0, 0, 0), 1)  # black outline
                    cv2.putText(frame, str(i), (pt[0] + 8, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Blend overlay for transparency
        cv2.addWeighted(overlay, self._config.fill_alpha, frame,
                        1 - self._config.fill_alpha, 0, frame)
        return frame

    def move_vertex(self, zone_name: str, vertex_idx: int,
                    new_pos: Tuple[int, int]) -> bool:
        """Move a single vertex of a zone and persist."""
        zone = self._zones.get(zone_name)
        if not zone or vertex_idx < 0 or vertex_idx >= len(zone.points):
            return False
        zone.points[vertex_idx] = new_pos
        self._save()
        return True

    def find_nearest_vertex(self, zone_name: str, point: Tuple[int, int],
                            max_dist: float = 20.0) -> Optional[int]:
        """Find the nearest vertex index in a zone within max_dist pixels."""
        zone = self._zones.get(zone_name)
        if not zone:
            return None
        best_idx, best_d = None, max_dist
        for i, vt in enumerate(zone.points):
            d = math.hypot(vt[0] - point[0], vt[1] - point[1])
            if d < best_d:
                best_d = d
                best_idx = i
        return best_idx

    def add_vertex(self, zone_name: str, position: Tuple[int, int]) -> bool:
        """Add a new vertex to the end of a zone's polygon and persist."""
        zone = self._zones.get(zone_name)
        if not zone:
            return False
        zone.points.append(position)
        self._save()
        return True

    def remove_vertex(self, zone_name: str, vertex_idx: int) -> bool:
        """Remove a vertex from a zone (must keep >= min_points)."""
        zone = self._zones.get(zone_name)
        if not zone or len(zone.points) <= self._config.min_points:
            return False
        if 0 <= vertex_idx < len(zone.points):
            zone.points.pop(vertex_idx)
            self._save()
            return True
        return False


class BoundaryDrawer:
    """
    Interactive polygon drawing tool using OpenCV mouse callbacks.

    Usage:
        drawer = BoundaryDrawer(frame, config)
        points = drawer.run()          # blocks until user confirms
    """

    def __init__(self, frame: np.ndarray, config: BoundaryConfig,
                 zone_name: str = "zone_1", window_name: Optional[str] = None):
        self._original = frame.copy()
        self._config = config
        self._zone_name = zone_name
        self._points: List[Tuple[int, int]] = []
        self._done = False
        # Reuse an existing window name if provided, otherwise create a new one
        self._window = window_name or "Draw Boundary"
        self._owns_window = window_name is None  # we created it, we destroy it

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self._done:
            self._points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last point
            if self._points:
                self._points.pop()

    def _draw_preview(self) -> np.ndarray:
        canvas = self._original.copy()
        if len(self._points) > 1:
            pts = np.array(self._points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False,
                          color=self._config.line_color,
                          thickness=self._config.line_thickness)
        for pt in self._points:
            cv2.circle(canvas, pt, self._config.point_radius,
                       self._config.point_color, -1)

        # HUD
        cv2.putText(canvas, f"Points: {len(self._points)} | LClick=add  RClick=undo  C=confirm  R=reset  Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return canvas

    def run(self) -> Optional[List[Tuple[int, int]]]:
        """Open an interactive window and return the polygon points, or None if cancelled."""
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._window, self._mouse_callback)

        # Show the initial frame so the window is visible and focusable
        cv2.imshow(self._window, self._draw_preview())
        cv2.waitKey(1)  # force window to render

        while True:
            canvas = self._draw_preview()
            cv2.imshow(self._window, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("c"):
                if len(self._points) >= self._config.min_points:
                    self._done = True
                    if self._owns_window:
                        cv2.destroyWindow(self._window)
                    return self._points
                else:
                    logger.warning("Need at least %d points", self._config.min_points)
            elif key == ord("r"):
                self._points.clear()
            elif key == ord("q") or key == 27:  # q or ESC
                if self._owns_window:
                    cv2.destroyWindow(self._window)
                return None

        return None
