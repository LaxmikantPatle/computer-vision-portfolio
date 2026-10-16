from ultralytics import YOLO
import argparse, cv2
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Validate and export evaluation results for fine-tuned YOLO")
    p.add_argument("--weights", required=True, help="best.pt path")
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--save", action="store_true", help="save validation plots/metrics if available")
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, device=args.device, imgsz=args.imgsz, conf=args.conf, iou=args.iou, plots=True, save_json=False)
    print(metrics)
    print("mAP50:", getattr(metrics, 'map50', 'N/A'))
    print("mAP50-95:", getattr(metrics, 'map', 'N/A'))
    print("precision:", getattr(metrics, 'precision', 'N/A'))
    print("recall:", getattr(metrics, 'recall', 'N/A'))
    print("f1:", getattr(metrics.box, 'f1', 'N/A') if hasattr(metrics, 'box') else 'N/A')

if __name__ == "__main__":
    main()
