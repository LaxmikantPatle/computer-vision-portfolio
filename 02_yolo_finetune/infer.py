from ultralytics import YOLO
import argparse, cv2, sys
from pathlib import Path
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Run inference on images/videos with fine-tuned weights")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--save", default=None, help="save video")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, stream=True)
    # handle single image case quickly
    from ultralytics.engine.results import Results
    # simple path: if image
    src = args.source
    if not src.isdigit() and Path(src).suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}:
        r = model.predict(src, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device)[0]
        im = r.plot()
        cv2.imwrite('predicted.jpg', im)
        if not args.no_show:
            cv2.imshow('pred', im); cv2.waitKey(0); cv2.destroyAllWindows()
        return
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    writer = None
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        r = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        out = r.plot()
        if writer is None and args.save:
            h,w = out.shape[:2]
            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w,h))
        if writer: writer.write(out)
        if not args.no_show:
            cv2.imshow('pred', out)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    if writer: writer.release()
    if not args.no_show: cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
