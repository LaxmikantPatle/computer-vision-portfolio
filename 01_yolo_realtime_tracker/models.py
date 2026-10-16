from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    cls_id: int
    cls_name: str
    conf: float
    track_id: Optional[int] = None

    @property
    def foot(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, y2

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class CrossingEvent:
    track_id: int
    cls_name: str
    direction: str
    position: Tuple[float, float]
    timestamp: float
    frame_index: int = 0


@dataclass
class FrameStats:
    live_counts: dict = field(default_factory=dict)
    total_counts: dict = field(default_factory=dict)
    fps: float = 0.0
    num_detections: int = 0
