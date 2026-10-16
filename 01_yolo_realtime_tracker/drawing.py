from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from line_crossing import LineCrossingCounter
from models import Detection

PALETTE = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (168, 153, 44),
    (255, 194, 0),
    (147, 69, 52),
    (255, 115, 100),
    (236, 24, 0),
    (255, 56, 132),
    (133, 0, 82),
    (255, 56, 203),
    (199, 55, 255),
    (180, 128, 255),
]


def class_color(cls_id: int) -> Tuple[int, int, int]:
    return PALETTE[cls_id % len(PALETTE)]


def _panel(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.45) -> None:
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return
    bg = np.full_like(roi, 20)
    cv2.addWeighted(bg, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.xyxy]
        color = class_color(det.cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tid = f"#{det.track_id}" if det.track_id is not None else ""
        label = f"{det.cls_name} {tid} {det.conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1, th + 6)
        cv2.rectangle(frame, (x1, ty - th - baseline - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        fx, fy = det.foot
        cv2.circle(frame, (int(fx), int(fy)), 3, color, -1)


def draw_crossing_line(
    frame: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    counter: LineCrossingCounter,
) -> None:
    a = (int(p1[0]), int(p1[1]))
    b = (int(p2[0]), int(p2[1]))
    for i in range(-1, 2):
        cv2.line(frame, a, b, (0, 215, 255), 2, cv2.LINE_AA, shift=0)
    cv2.putText(
        frame,
        "COUNT LINE",
        (a[0] + 8, max(a[1] - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )
    for ev in list(counter.events)[-8:]:
        cx, cy = int(ev.position[0]), int(ev.position[1])
        age = int((datetime.now().timestamp() - ev.timestamp) * 2)
        radius = max(4, 14 - age)
        cv2.circle(frame, (cx, cy), radius, (0, 255, 128), 2, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    live_counts: Dict[str, int],
    total_counts: Dict[str, int],
    counter: LineCrossingCounter,
    fps: float,
    num_detections: int,
    source_label: str,
    max_rows: int = 9,
) -> None:
    h, w = frame.shape[:2]
    panel_w = 290
    rows = min(len(total_counts), max_rows) + 3 + len(counter.events) // 3
    panel_h = min(h - 10, 70 + 22 * max_rows)
    _panel(frame, 8, 8, panel_w, panel_h)

    y = 30
    cv2.putText(
        frame,
        f"FPS: {fps:5.1f}   Dets: {num_detections}",
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (80, 255, 80),
        2,
        cv2.LINE_AA,
    )
    y += 24
    cv2.putText(
        frame,
        f"Source: {source_label}",
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    y += 24
    cv2.putText(
        frame,
        "LIVE (total, unique IDs)",
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    y += 20
    for cls_name in sorted(total_counts, key=lambda c: -total_counts[c])[:max_rows]:
        color = class_color(abs(hash(cls_name)) % len(PALETTE))
        live = live_counts.get(cls_name, 0)
        total = total_counts[cls_name]
        text = f"{cls_name:<12} now {live:>3}  all {total:>4}"
        cv2.putText(
            frame,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 20
    y += 6
    cv2.putText(
        frame,
        f"CROSSINGS: {counter.total()}",
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )
    y += 20
    for ev in list(counter.events)[-6:]:
        clock = datetime.fromtimestamp(ev.timestamp).strftime("%H:%M:%S")
        text = f"{ev.cls_name} #{ev.track_id} {ev.direction} {clock}"
        cv2.putText(
            frame,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 17


def draw_fps(frame: np.ndarray, fps: float) -> None:
    h, w = frame.shape[:2]
    label = f"{fps:.1f} FPS"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    x = w - tw - 16
    cv2.rectangle(frame, (x - 8, 8), (w - 8, 8 + th + 14), (20, 20, 20), -1)
    cv2.putText(
        frame,
        label,
        (x, 8 + th + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (80, 255, 80),
        2,
        cv2.LINE_AA,
    )
