import math
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from models import CrossingEvent


class LineCrossingCounter:
    def __init__(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        history: int = 40,
        max_point_jump: float = 250.0,
    ):
        self.p1 = p1
        self.p2 = p2
        self.max_point_jump = max_point_jump
        self._prev_points: Dict[int, Tuple[float, float]] = {}
        self.events: deque = deque(maxlen=history)
        self.counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"up": 0, "down": 0, "left": 0, "right": 0, "total": 0}
        )
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        self.orientation = "horizontal" if abs(dx) >= abs(dy) else "vertical"

    def _signed(self, point: Tuple[float, float]) -> float:
        return (
            (self.p2[0] - self.p1[0]) * (point[1] - self.p1[1])
            - (self.p2[1] - self.p1[1]) * (point[0] - self.p1[0])
        )

    def _crossing_point(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        sa: float,
        sb: float,
    ) -> Tuple[float, float]:
        denom = sa - sb
        t = 0.5 if abs(denom) < 1e-9 else abs(sa) / denom
        t = max(0.0, min(1.0, t))
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    def _direction(self, a: Tuple[float, float], b: Tuple[float, float]) -> str:
        if self.orientation == "horizontal":
            return "down" if b[1] > a[1] else "up"
        return "right" if b[0] > a[0] else "left"

    def update(
        self,
        track_id: Optional[int],
        cls_name: str,
        point: Tuple[float, float],
        frame_index: int = 0,
    ) -> Optional[CrossingEvent]:
        if track_id is None:
            return None
        prev = self._prev_points.get(track_id)
        self._prev_points[track_id] = point
        if prev is None:
            return None
        jump = math.dist(prev, point)
        if jump > self.max_point_jump:
            return None
        sa = self._signed(prev)
        sb = self._signed(point)
        if sa == 0 or sb == 0 or (sa > 0) == (sb > 0):
            return None
        direction = self._direction(prev, point)
        position = self._crossing_point(prev, point, sa, sb)
        event = CrossingEvent(
            track_id=track_id,
            cls_name=cls_name,
            direction=direction,
            position=position,
            timestamp=time.time(),
            frame_index=frame_index,
        )
        self.counts[cls_name][direction] += 1
        self.counts[cls_name]["total"] += 1
        self.events.append(event)
        return event

    def prune(self, active_track_ids: set) -> None:
        stale = [tid for tid in self._prev_points if tid not in active_track_ids]
        for tid in stale:
            del self._prev_points[tid]

    def total(self, cls_name: Optional[str] = None) -> int:
        if cls_name is not None:
            return self.counts[cls_name]["total"]
        return sum(c["total"] for c in self.counts.values())

    def summary(self) -> List[dict]:
        rows = []
        for cls_name, c in sorted(self.counts.items()):
            rows.append(
                {
                    "class": cls_name,
                    "total": c["total"],
                    "up": c["up"],
                    "down": c["down"],
                    "left": c["left"],
                    "right": c["right"],
                }
            )
        return rows
