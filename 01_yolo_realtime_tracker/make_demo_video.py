import argparse
from pathlib import Path

import cv2
import numpy as np
import requests

SAMPLE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg"


def fetch_bus_image() -> np.ndarray:
    cache = Path("bus.jpg")
    if cache.exists():
        img = cv2.imread(str(cache))
        if img is not None:
            return img
    resp = requests.get(SAMPLE_URL, timeout=30)
    resp.raise_for_status()
    cache.write_bytes(resp.content)
    img = cv2.imread(str(cache))
    if img is None:
        raise RuntimeError("downloaded image could not be decoded")
    return img


def make_panning_video(img: np.ndarray, out_path: str, frames: int = 180, fps: int = 25) -> None:
    h, w = img.shape[:2]
    canvas_h = int(h * 1.9)
    canvas = np.zeros((canvas_h, w, 3), dtype=np.uint8)
    canvas[:h] = img
    canvas[h:] = cv2.resize(img, (w, canvas_h - h))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    max_shift = canvas_h - h
    for i in range(frames):
        t = i / max(frames - 1, 1)
        shift = int(max_shift * (0.5 - 0.5 * np.cos(2 * np.pi * t)))
        writer.write(canvas[shift : shift + h])
    writer.release()
    print(f"wrote {out_path} ({frames} frames). Objects scroll down then up across the count line.")


def make_synthetic_video(out_path: str, frames: int = 200, fps: int = 25, n_objects: int = 6) -> None:
    w, h = 960, 540
    rng = np.random.default_rng(7)
    colors = [(56, 56, 255), (0, 200, 255), (80, 220, 80), (255, 120, 80), (200, 80, 255), (255, 200, 0)]
    objects = []
    for i in range(n_objects):
        objects.append(
            {
                "x": rng.uniform(0.1, 0.9) * w,
                "y": rng.uniform(-0.2, 0.5) * h,
                "vy": rng.uniform(2.5, 5.5),
                "size": (int(rng.uniform(50, 90)), int(rng.uniform(90, 150))),
                "color": colors[i % len(colors)],
            }
        )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for _ in range(frames):
        frame = np.full((h, w, 3), 38, dtype=np.uint8)
        for i in range(0, h, 40):
            cv2.line(frame, (0, i), (w, i), (48, 48, 48), 1)
        for obj in objects:
            obj["y"] += obj["vy"]
            if obj["y"] > h + obj["size"][1]:
                obj["y"] = -obj["size"][1]
                obj["x"] = float(rng.uniform(0.1, 0.9) * w)
            bw, bh = obj["size"]
            x1 = int(obj["x"] - bw / 2)
            y1 = int(obj["y"] - bh / 2)
            cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), obj["color"], -1)
            cv2.circle(frame, (int(obj["x"]), y1 - 18), 16, obj["color"], -1)
        writer.write(frame)
    writer.release()
    print(f"wrote {out_path} (synthetic shapes, useful for testing the overlay/counting pipeline only)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build demo videos for the tracker")
    p.add_argument("--out", default="demo_video.mp4")
    p.add_argument("--synthetic", action="store_true", help="offline synthetic moving-shapes video")
    p.add_argument("--frames", type=int, default=180)
    args = p.parse_args()
    if args.synthetic:
        make_synthetic_video(args.out, frames=args.frames)
    else:
        try:
            img = fetch_bus_image()
            make_panning_video(img, args.out, frames=args.frames)
        except Exception as exc:
            print(f"download failed ({exc}); falling back to synthetic video")
            make_synthetic_video(args.out, frames=args.frames)


if __name__ == "__main__":
    main()
