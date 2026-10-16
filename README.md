# OpenCV / YOLO Portfolio Projects

Six self-contained computer-vision projects aimed at an **Applied CV / CV+LLM** role.
Each folder has its own `README.md` with setup, commands, and interview talking points.

| # | Project | Stack | Highlights |
|---|---------|-------|------------|
| 1 | [Real-time YOLO tracker](01_yolo_realtime_tracker/) | YOLO11, OpenCV, ByteTrack | Detection, per-class counts, persistent IDs, line-crossing counter, FPS HUD |
| 2 | [YOLO fine-tune](02_yolo_finetune/) | YOLO11, Colab GPU | Custom helmet/no-helmet dataset, YOLO txt labels, train/val/test split, mAP@50 + confusion matrix |
| 3 | [FastAPI + Docker API](03_fastapi_yolo_docker/) | FastAPI, Docker | `POST /detect` image → JSON boxes/classes/confidences |
| 4 | [CV + LLM insights](04_cv_llm_insights/) | YOLO11, OpenAI | Detections → text counts → LLM incident/situation summary |
| 5 | [CLIP + FAISS visual search](05_clip_faiss_visual_search/) | CLIP, FAISS | Text→image semantic search over an embedding index |
| 6 | [Document scanner](06_opencv_document_scanner/) | OpenCV (classical) | Canny + contours + perspective transform + adaptive threshold |

## Recommended order

Tier 1 (must-do): **1 → 2**. Tier 2 (if time): **3 → 4**. Tier 3 (stretch): **5** or **6**.

## Environment

Built and tested with Python 3.11. Install per project with its `requirements.txt`:

```powershell
pip install -r 01_yolo_realtime_tracker\requirements.txt
```

Quick start (Project 1, no webcam needed):

```powershell
cd 01_yolo_realtime_tracker
python make_demo_video.py --out demo_video.mp4
python main.py --source demo_video.mp4 --save annotated_demo.mp4
```

## Core concepts you can now explain from experience

- **How YOLO works** (one-stage, anchor-free, DFL/CIoU losses) - Project 1 README
- **NMS & IoU** (`--conf`/`--iou`) - Project 1 README
- **Tracking-by-detection / ByteTrack** - Project 1 README
- **Datasets, YOLO txt format, splits, augmentation, overfitting, model size** - Project 2 README
- **Serving & containers** - Project 3 README
- **CV + LLM / GenAI blend** - Project 4 README
- **Embeddings, vector DBs, multimodal models** - Project 5 README
- **Classical CV fundamentals** - Project 6 README
