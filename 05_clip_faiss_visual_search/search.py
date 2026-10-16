import argparse, os
from clip_faiss import ClipIndexer
from pathlib import Path
import cv2

def parse_args():
    p = argparse.ArgumentParser(description="CLIP + FAISS visual search (text -> similar images)")
    p.add_argument("--index", action="store_true", help="build index from --images")
    p.add_argument("--images", default="images", help="image folder to index")
    p.add_argument("--prefix", default="index")
    p.add_argument("--query", default=None, help="text query e.g. 'red car', 'person with backpack'")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--show", action="store_true", help="show top results with OpenCV")
    return p.parse_args()

def main():
    args = parse_args()
    ci = ClipIndexer(device=args.device)
    if args.index:
        ci.build_index(args.images, out_prefix=args.prefix)
        return
    ci.load_index(args.prefix)
    if not args.query:
        print("no query given; use --query 'text'")
        return
    res = ci.search(args.query, k=args.k)
    for i, (p, s) in enumerate(res):
        print(f"{i+1}. {s:.3f}  {p}")
    if args.show:
        for i, (p, s) in enumerate(res[:3]):
            img = cv2.imread(p)
            if img is not None:
                cv2.imshow(f"#{i+1} {Path(p).name}", img)
        cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
