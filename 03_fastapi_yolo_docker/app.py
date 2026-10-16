from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
from typing import List, Dict, Any
import time

app = FastAPI(title="YOLO Detection API", version="1.0.0", description="POST /detect returns boxes/classes/confidences")
model = YOLO("yolo11n.pt")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "yolo11n.pt"}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.35, iou: float = 0.45, imgsz: int = 640):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only image uploads allowed")
    try:
        data = await file.read()
        img = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image")
        t0 = time.perf_counter()
        res = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        dt = time.perf_counter() - t0
        boxes = res.boxes
        out: List[Dict[str, Any]] = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            cls = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            names = res.names
            for b, c, cf in zip(xyxy, cls, confs):
                out.append({
                    "x1": round(b[0], 2),
                    "y1": round(b[1], 2),
                    "x2": round(b[2], 2),
                    "y2": round(b[3], 2),
                    "class_id": int(c),
                    "class": names[int(c)],
                    "confidence": round(float(cf), 4),
                })
        return JSONResponse({
            "count": len(out),
            "inference_ms": round(dt * 1000, 2),
            "detections": out,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"detection failed: {e}")
