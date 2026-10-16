from pathlib import Path
import argparse, random, shutil, sys

YOLO_TXT = "*.txt"


def parse_args():
    p = argparse.ArgumentParser(description="Prepare YOLO dataset split from images+labels (YOLO txt)")
    p.add_argument("--src", type=str, required=True, help="folder containing images/ and labels/")
    p.add_argument("--dst", type=str, default="dataset", help="output root (will create dataset/images|labels/{train,val,test})")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", help="copy files instead of symlink (safer on Windows)")
    return p.parse_args()


def copy_or_link(src, dst, copy=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy or sys.platform == "win32":
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)


def main():
    args = parse_args()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    imgs = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
        imgs.extend(src.rglob("*"+ext))
    imgs = sorted(set(imgs))
    if not imgs:
        sys.exit(f"no images found in {src}")
    labels = [p.with_suffix(".txt") for p in imgs]
    valid = [(i, l) for i, l in zip(imgs, labels) if l.exists()]
    if len(valid) < len(imgs):
        print("warning: some images missing YOLO .txt labels; only keeping paired")
    imgs, labels = zip(*valid) if valid else ([], [])
    imgs = list(imgs); labels = list(labels)
    random.seed(args.seed)
    combined = list(zip(imgs, labels))
    random.shuffle(combined)
    n = len(combined)
    v = int(n * args.val)
    t = int(n * args.test)
    splits = {
        "train": combined[v + t:],
        "val": combined[:v],
        "test": combined[v:v + t],
    }
    for split, items in splits.items():
        for i, (img, lbl) in enumerate(items):
            name = img.stem
            copy_or_link(img, dst / "images" / split / f"{name}{img.suffix}", copy=args.copy)
            copy_or_link(lbl, dst / "labels" / split / f"{name}.txt", copy=args.copy)
    print(f"split done: train {len(splits['train'])} val {len(splits['val'])} test {len(splits['test'])} -> {dst}")
    print("update data.yaml path: train: images/train  val: images/val  test: images/test  nc/names")


if __name__ == "__main__":
    main()
