# Project 1 - Real-Time Object Detection, Tracking & Line-Crossing Counter (YOLO11 + OpenCV)

A production-style real-time vision pipeline built on **Ultralytics YOLO11** and **OpenCV**.
It detects objects, gives every object a persistent ID (ByteTrack), counts objects per class,
counts objects crossing a virtual line, and draws an FPS/counter HUD over the video.

## Features

- Detection + custom logic: colored boxes, class + confidence + track ID labels.
- Multi-object tracking with `model.track(..., tracker="bytetrack.yaml")` (persistent IDs).
- Per-class **live counts** and **unique-object totals** (via track-ID sets).
- **Line-crossing counter** with direction (up/down/left/right) and crossing-point markers.
- Smoothed **FPS** overlay (EMA) + headless-safe rendering.
- Works on a **webcam**, a **video file**, or a single **image**.
- Optional annotated video export (`--save`) and live snapshot with `s`.

## Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, capture loop, wiring everything together |
| `line_crossing.py` | `LineCrossingCounter` - signed-distance side test + direction logic |
| `drawing.py` | Boxes, HUD panel, count line, crossing markers, FPS |
| `models.py` | `Detection` / `CrossingEvent` dataclasses |
| `make_demo_video.py` | Builds a test video (real sample image pan, or synthetic) |

## Install

```bash
pip install -r requirements.txt
```

The first run downloads `yolo11n.pt` (~5 MB) and auto-installs the tracker dependency `lap`.

## Run

```bash
# Webcam (index 0)
python main.py --source 0

# Video file, count only people (COCO class 0) and cars (2)
python main.py --source street.mp4 --classes 0 2

# Custom counting line (x1,y1,x2,y2)
python main.py --source street.mp4 --line 100,300,900,300

# Save annotated output and run without a GUI window
python main.py --source street.mp4 --no-show --save out.mp4

# Single image
python main.py --source bus.jpg
```

Key controls in the window: **q** quit, **s** save snapshot.

### Quick demo without a webcam

```bash
python make_demo_video.py --out demo_video.mp4   # downloads a sample image and pans it
python main.py --source demo_video.mp4 --save annotated_demo.mp4
```

## How it works

1. Each frame goes through `model.track(...)` which runs the YOLO detector then the ByteTrack
   association stage, returning `boxes.xyxy`, `boxes.cls`, `boxes.conf`, and `boxes.id`.
2. Every box becomes a `Detection`. The foot point (bottom-center) of each box is fed to the
   `LineCrossingCounter`.
3. The counter computes the **signed distance** of the foot point to the line from two consecutive
   frames. A sign flip means the object crossed. Direction comes from the movement vector, and the
   exact crossing point is interpolated for the marker.
4. Counts, unique IDs, events and FPS are drawn as an HUD.

## Interview notes

**How does YOLO work?**
A single CNN predicts, in one forward pass, a grid of cells; each cell regresses several bounding
boxes (x, y, w, h, objectness) and class probabilities. YOLO11 is anchor-free: each location predicts
a box with a distribution over the four edges (DFL), trained with a classification loss, a
box-regression loss (CIoU) and a distribution focal loss. Because it is a one-stage, single-shot
detector it is fast enough for real time.

**What is IoU?**
Intersection over Union = area of overlap / area of union of two boxes. It measures how well a
predicted box matches a ground-truth (or another predicted) box. The `--iou` flag is the NMS
threshold on IoU.

**What is NMS?**
Non-Maximum Suppression removes duplicate boxes for the same object: sort candidate boxes by
confidence, keep the best, delete any remaining box whose IoU with a kept box exceeds the threshold,
repeat. `--conf 0.35 --iou 0.45` control this.

**Why ByteTrack?**
It is a tracking-by-detection method. It matches high-confidence detections to existing tracks using
IoU/motion, and crucially also uses *low-confidence* boxes in a second association pass to keep
IDs stable through occlusion. That is why `model.track()` gives stable `#id` values.
