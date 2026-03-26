"""
Visualization utilities — overlays, HUD, bounding boxes, warnings.
"""

import cv2
import numpy as np
from typing import List

from config import DisplayConfig
from detector import Detection


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    config: DisplayConfig,
) -> np.ndarray:
    """Draw bounding boxes and labels for all detections."""
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        label = f"Person {det.confidence:.0%}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                       config.bbox_color, config.bbox_thickness)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       config.label_font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1),
                       config.bbox_color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, config.label_font_scale,
                    config.label_color, 1, cv2.LINE_AA)

        # Reference point
        cv2.circle(frame, det.bottom_center, 4, (0, 255, 255), -1)

    return frame


def draw_intrusion_warning(
    frame: np.ndarray,
    intrusions: List[dict],
    config: DisplayConfig,
) -> np.ndarray:
    """Overlay flashing intrusion warning when intrusions are active."""
    if not intrusions:
        return frame

    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Red tint border
    cv2.rectangle(overlay, (0, 0), (w, h), config.warning_color, 15)

    # Warning text
    total_persons = sum(i["num_persons"] for i in intrusions)
    zone_names = ", ".join(i["zone_name"] for i in intrusions)
    text = f"INTRUSION ALERT — {total_persons} person(s) in [{zone_names}]"

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                   config.warning_font_scale, 2)
    cx = (w - tw) // 2
    cy = 50

    # Background bar
    cv2.rectangle(overlay, (0, cy - th - 15), (w, cy + 10),
                   (0, 0, 180), -1)
    cv2.putText(overlay, text, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, config.warning_font_scale,
                (255, 255, 255), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    inference_ms: float,
    num_detections: int,
    num_zones: int,
    monitoring_active: bool,
    config: DisplayConfig,
) -> np.ndarray:
    """Draw heads-up display with performance metrics."""
    if not config.show_fps:
        return frame

    lines = [
        f"FPS: {fps:.1f}",
        f"Inference: {inference_ms:.1f} ms",
        f"Persons: {num_detections}",
        f"Zones: {num_zones}",
        f"Monitoring: {'ON' if monitoring_active else 'OFF'}",
    ]

    y = frame.shape[0] - 20
    for line in reversed(lines):
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        y -= 22

    return frame
