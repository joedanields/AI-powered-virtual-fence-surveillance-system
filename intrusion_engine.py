"""
Intrusion engine — ties detection + boundary + alert + database together.

Implements anti-spam logic, time-based scheduling, and
per-zone intrusion state management.
"""

import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import IntrusionConfig, ScheduleConfig
from detector import Detection
from boundary import BoundaryManager
from alert_system import AlertDispatcher, VideoClipWriter
from database import IntrusionDatabase

logger = logging.getLogger(__name__)


class IntrusionEngine:
    """
    Core intrusion decision engine.

    For every frame:
      1. Maps each Detection to its reference point (center / bottom_center).
      2. Tests each point against every boundary zone.
      3. Applies cooldown to suppress duplicate alerts.
      4. Dispatches alerts and logs to DB when intrusion is confirmed.
    """

    def __init__(
        self,
        config: IntrusionConfig,
        schedule_config: ScheduleConfig,
        boundary_mgr: BoundaryManager,
        alert_dispatcher: AlertDispatcher,
        database: IntrusionDatabase,
        camera_id: str = "cam_01",
    ):
        self._config = config
        self._schedule = schedule_config
        self._boundary = boundary_mgr
        self._alerts = alert_dispatcher
        self._db = database
        self._camera_id = camera_id

        # Anti-spam: zone_name → last_alert_timestamp
        self._last_alert: Dict[str, float] = defaultdict(float)

        # Active video clip writers: zone_name → VideoClipWriter
        self._clip_writers: Dict[str, VideoClipWriter] = {}

        # Per-frame state (reset every process() call)
        self._current_intrusions: List[dict] = []

    # ── public interface ───────────────────────
    def process(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> List[dict]:
        """
        Evaluate detections against boundary zones.

        Returns list of intrusion dicts (one per triggered zone) with keys:
            zone_name, num_persons, alerted (bool)
        """
        self._current_intrusions = []

        # Check if monitoring is active (time schedule)
        if not self._is_monitoring_active():
            return self._current_intrusions

        if not self._boundary.has_zones():
            logger.debug("No boundary zones configured — skipping intrusion check")
            return self._current_intrusions

        if detections:
            logger.debug("Processing %d detection(s) against %d zone(s)",
                         len(detections), len(self._boundary.get_all_zones()))

        # Group detections by violated zone
        zone_persons: Dict[str, List[Detection]] = defaultdict(list)

        for det in detections:
            ref_points = self._reference_points(det)
            violated_zones_set: set = set()
            for pt in ref_points:
                for zone_name in self._boundary.check_point(pt):
                    violated_zones_set.add(zone_name)
            if not violated_zones_set:
                logger.debug(
                    "Detection bbox=%s ref_points=%s matched NO zones",
                    det.bbox, ref_points,
                )
            for zone_name in violated_zones_set:
                zone_persons[zone_name].append(det)

        # Process each violated zone
        now = time.time()
        for zone_name, persons in zone_persons.items():
            should_alert = self._should_alert(zone_name, now)
            entry = {
                "zone_name": zone_name,
                "num_persons": len(persons),
                "alerted": should_alert,
                "detections": persons,
            }
            self._current_intrusions.append(entry)

            if should_alert:
                self._last_alert[zone_name] = now
                result = self._alerts.trigger(
                    frame=frame,
                    camera_id=self._camera_id,
                    zone_name=zone_name,
                    num_persons=len(persons),
                )
                # Log to DB
                self._db.log_event(
                    camera_id=self._camera_id,
                    screenshot_path=result.get("screenshot_path", ""),
                    video_path=result.get("video_path", ""),
                    zone_name=zone_name,
                    num_persons=len(persons),
                )

                # Start video clip recording
                clip_writer = self._alerts.start_video_clip(
                    camera_id=self._camera_id,
                    resolution=(frame.shape[1], frame.shape[0]),
                )
                if clip_writer:
                    self._clip_writers[zone_name] = clip_writer

        # Feed active clip writers
        self._update_clip_writers(frame)

        return self._current_intrusions

    @property
    def current_intrusions(self) -> List[dict]:
        return self._current_intrusions

    # ── internals ──────────────────────────────
    def _reference_points(self, det: Detection) -> List[Tuple[int, int]]:
        """Return multiple test points for a detection to improve zone overlap."""
        x1, y1, x2, y2 = det.bbox
        points = [
            det.center,
            det.bottom_center,
            ((x1 + x2) // 2, y1),        # top-center
            (x1, (y1 + y2) // 2),        # mid-left
            (x2, (y1 + y2) // 2),        # mid-right
        ]
        return points

    def _should_alert(self, zone_name: str, now: float) -> bool:
        """Anti-spam: only alert once per cooldown period per zone."""
        last = self._last_alert.get(zone_name, 0.0)
        return (now - last) >= self._config.cooldown_seconds

    def _is_monitoring_active(self) -> bool:
        """Check time-based schedule. Returns True if monitoring should run."""
        if not self._schedule.enabled:
            return True  # always active when schedule is off

        now = datetime.now().time()
        start = datetime.strptime(self._schedule.active_start, "%H:%M").time()
        end = datetime.strptime(self._schedule.active_end, "%H:%M").time()

        if start <= end:
            return start <= now <= end
        else:
            # Overnight range (e.g. 22:00→06:00)
            return now >= start or now <= end

    def _update_clip_writers(self, frame: np.ndarray):
        """Feed frames to active clip writers and finalize completed ones."""
        done_zones = []
        for zone_name, writer in self._clip_writers.items():
            writer.write(frame)
            if writer.is_done:
                writer.release()
                done_zones.append(zone_name)
        for z in done_zones:
            del self._clip_writers[z]
