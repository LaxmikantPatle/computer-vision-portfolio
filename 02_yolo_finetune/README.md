# Project 2 - Fine-Tune YOLO on a Small Custom Dataset (Helmet vs No-Helmet)

Train YOLO11 on a custom dataset (YOLO annotation format `.txt`), evaluate with mAP@50/Precision/Recall + confusion matrix, and run inference on images/videos. Includes a Colab notebook for GPU training.

## Dataset choice

**Helmet vs No-Helmet** (safety/compliance) - easy to label, visually clear, common in CV interviews. You can source a ready dataset from [Roboflow Universe](https://universe.roboflow.com/) (search “helmet detection”, “hard hat”, “helmet no helmet”) or annotate a few hundred images.

## Folder layout

After preparing/splitting, the expected structure (relative to this folder):

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
data.yaml  (points to dataset/)
```

`data.yaml`:
```yaml
path: ../dataset        # or absolute path to dataset root
train: images/train
val: images/val
test: images/test
nc: 2
names: ['helmet', 'no-helmet']
```

## 1) Prepare a dataset

### Option A: Download from Roboflow Universe

1. Go to [Roboflow Universe](https://universe.roboflow.com/search?q=helmet%20no%20helmet) and pick a dataset (check license).
2. Export as **YOLOv11** format (downloads a ZIP containing `data.yaml`, `train/val/test` images+labels). Extract it so `data.yaml` is at repo root or point `--data` to it.

### Option B: Split your own collected images

If you already have images + YOLO `.txt` labels in one folder (`img1.jpg, img1.txt,...`):

```bash
python prepare_split.py --src "C:/path/to/raw" --dst dataset --val 0.15 --test 0.15 --copy
```

Notes:
- YOLO txt format: `class_id x_center y_center width height` (normalized 0–1).
- Classes are 0 = helmet, 1 = no-helmet in this template.
- Use `--copy` on Windows if symlinks are blocked.

## 2) Train (GPU recommended - Colab)

Local CPU is slow for larger epochs. Use the included Colab notebook: `colab_finetune_yolo11_helmet.ipynb`.

Local training (if GPU available):

```bash
python train.py --data data.yaml --model yolo11n.pt --epochs 40 --imgsz 640 --batch 16 --device 0
```

Suggested model sizes: **n** (nano) for speed/edge, **s** (small) for small datasets/better accuracy, **m** larger. For 30–50 epochs on ~500–2000 images, `yolo11n.pt` is a good starting point to avoid overfitting.

### What to watch (explain in interview)

- **Train/val split**: stratify by scenes, avoid leakage (same frame in train+val).
- **Annotation format**: YOLO txt (normalized), class IDs match `names`.
- **Augmentation**: Ultralytics applies Mosaic, MixUp, HSV, flips by default - helps small datasets, but too aggressive can hurt if classes are small/rare.
- **Overfitting**: if train loss keeps dropping but val mAP plateaus/declines while box/cls loss diverge, reduce epochs, increase regularization, or add more data/augmentation tuning.
- **Model size choice**: `n < s < m < l < x` - larger = more capacity + accuracy, slower + needs more data/GPU. For small custom datasets, start with **n** (less overfit).
- **Learning rate**: `lr0` (initial LR), cosine scheduler; `optimizer='auto'` is safe.

## 3) Evaluate

After training, best weights are at `runs/train/helmet_yolo11n/weights/best.pt` (or your `--project/--name`).

```bash
python eval.py --weights runs/train/helmet_yolo11n/weights/best.pt --data data.yaml
```

Look for:
- **mAP@50** (mean Average Precision at IoU>=0.5) - primary for detection
- **Precision**, **Recall** (tradeoff: high precision = fewer false positives, high recall = fewer false negatives)
- **Confusion matrix** (plots=True in val) - shows per-class confusion, helpful to debug misclassifications (helmet vs no-helmet often confused in occlusion/low light).

Ultralytics also saves plots under the run folder (e.g. `confusion_matrix.png`, `F1_curve.png`, `PR_curve.png`, loss curves).

## 4) Inference

Test on an image, video, or webcam:

```bash
python infer.py --weights runs/train/helmet_yolo11n/weights/best.pt --source test.jpg
python infer.py --weights runs/train/helmet_yolo11n/weights/best.pt --source video.mp4 --save annotated.mp4
python infer.py --weights runs/train/helmet_yolo11n/weights/best.pt --source 0 --no-show
```

## Tips

- Keep a small **test set** untouched (separately evaluated) to report final numbers honestly.
- If classes are imbalanced, watch precision/recall per class and consider class weights/data balancing.
- For safety-critical (helmets), prefer higher **recall** (don’t miss missing helmets) and tune confidence accordingly.
