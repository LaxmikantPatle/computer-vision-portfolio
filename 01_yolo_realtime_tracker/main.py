import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from drawing import draw_crossing_line, draw_detections, draw_fps, draw_hud
from line_crossing import LineCrossingCounter
from models import Detection

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time YOLO detection, tracking, line-crossing counter")
    p.add_argument("--source", default="0", help="webcam index (0), video file, or image path")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO weights (yolo11n/s/m.pt or custom)")
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    p.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference size")
    p.add_argument("--classes", type=int, nargs="*", default=None, help="restrict to COCO class ids (0=person, 2=car, 3=bike...)")
    p.add_argument("--device", default=None, help="cpu, 0, 0,1 ...")
    p.add_argument("--line", default=None, help="counting line as x1,y1,x2,y2 in pixels")
    p.add_argument("--line-frac", type=float, default=0.55, help="auto line y-position as fraction of frame height")
    p.add_argument("--save", default=None, help="save annotated video to this path")
    p.add_argument("--no-show", action="store_true", help="disable GUI window")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = unlimited)")
    p.add_argument("--playback-fps", type=float, default=25.0, help="fps metadata for saved video")
    return p.parse_args()


def parse_source(source: str):
    if source.isdigit():
        return "webcam", int(source)
    path = Path(source)
    if not path.exists():
        sys.exit(f"source not found: {source}")
    if path.suffix.lower() in IMAGE_EXTS:
        return "image", path
    return "video", path


def parse_line(frame_shape, args):
    h, w = frame_shape[:2]
    if args.line:
        x1, y1, x2, y2 = [float(v) for v in args.line.split(",")]
        return (x1, y1), (x2, y2)
    y = int(h * args.line_frac)
    return (int(w * 0.1), y), (int(w * 0.9), y)


def boxes_to_detections(result, names) -> list:
    dets = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return dets
    xyxy = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    clss = boxes.cls.int().cpu().tolist()
    ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(xyxy)
    for box, conf, cls_id, track_id in zip(xyxy, confs, clss, ids):
        dets.append(
            Detection(
                xyxy=tuple(box),
                cls_id=int(cls_id),
                cls_name=names[int(cls_id)],
                conf=float(conf),
                track_id=int(track_id) if track_id is not None else None,
            )
        )
    return dets


def run_image(args, model, names) -> None:
    _, path = parse_source(args.source)
    frame = cv2.imread(str(path))
    if frame is None:
        sys.exit(f"could not read image: {path}")
    result = model.predict(
        frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        classes=args.classes, device=args.device, verbose=False,
    )[0]
    dets = boxes_to_detections(result, names)
    live = dict(Counter(d.cls_name for d in dets))
    p1, p2 = parse_line(frame.shape[:2], args)
    counter = LineCrossingCounter(p1, p2)
    for d in dets:
        counter.update(d.track_id, d.cls_name, d.foot)
    draw_detections(frame, dets)
    draw_crossing_line(frame, p1, p2, counter)
    draw_hud(frame, live, live, counter, 0.0, len(dets), str(path))
    cv2.imwrite("output_image.jpg", frame)
    print("saved annotated image -> output_image.jpg")
    print("counts:", live)
    if not args.no_show:
        cv2.imshow("detections", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_stream(args, model, names) -> None:
    kind, source = parse_source(args.source)
    cap = cv2.VideoCapture(source if kind == "webcam" else str(source))
    if not cap.isOpened():
        sys.exit(f"could not open {kind} source: {source}")

    ok, frame = cap.read()
    if not ok:
        sys.exit("could not read first frame")
    h, w = frame.shape[:2]
    p1, p2 = parse_line((h, w), args)
    counter = LineCrossingCounter(p1, p2)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.playback_fps, (w, h))

    show = not args.no_show
    ids_seen = defaultdict(set)
    ema_fps = 0.0
    frame_index = 0
    label = f"webcam {source}" if kind == "webcam" else Path(source).name

    try:
        while True:
            if frame_index > 0:
                ok, frame = cap.read()
                if not ok:
                    break
            t0 = time.perf_counter()

            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=args.classes,
                device=args.device,
                verbose=False,
            )
            dets = boxes_to_detections(results[0], names)

            for d in dets:
                if d.track_id is not None:
                    ids_seen[d.cls_name].add(d.track_id)
                    counter.update(d.track_id, d.cls_name, d.foot, frame_index)
            live_counts = dict(Counter(d.cls_name for d in dets))
            total_counts = {cls: len(ids) for cls, ids in ids_seen.items()}
            counter.prune({d.track_id for d in dets if d.track_id is not None})

            dt = time.perf_counter() - t0
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            ema_fps = inst_fps if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst_fps

            draw_detections(frame, dets)
            draw_crossing_line(frame, p1, p2, counter)
            draw_hud(frame, live_counts, total_counts, counter, ema_fps, len(dets), label)
            draw_fps(frame, ema_fps)

            if writer is not None:
                writer.write(frame)
            if show:
                try:
                    cv2.imshow("YOLO tracker - q quit, s snapshot", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("s"):
                        snap = f"snapshot_{frame_index:06d}.jpg"
                        cv2.imwrite(snap, frame)
                        print("saved", snap)
                except cv2.error:
                    print("no GUI available, continuing headless")
                    show = False

            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print("\n=== line crossing summary ===")
    rows = counter.summary()
    if not rows:
        print("no crossings recorded")
    for row in rows:
        print(row)
    print("unique objects seen:", total_counts)


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    kind, _ = parse_source(args.source)
    if kind == "image":
        run_image(args, model, names=model.names)
    else:
        run_stream(args, model, names=model.names)


if __name__ == "__main__":
    main()
