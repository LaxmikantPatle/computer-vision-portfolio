from ultralytics import YOLO
import argparse, os

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLO for helmet/no-helmet")
    p.add_argument("--data", default="data.yaml", help="path to data.yaml")
    p.add_argument("--model", default="yolo11n.pt", choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="cpu, 0, cuda etc")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="helmet_yolo11n")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--optimizer", default="auto", help="SGD/AdamW/auto")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--save", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        lr0=args.lr0,
        optimizer=args.optimizer,
        patience=args.patience,
        exist_ok=True,
        pretrained=True,
        val=True,
    )
    print("training finished; best weights:", results.best)

if __name__ == "__main__":
    main()
