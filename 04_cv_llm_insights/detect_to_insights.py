from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

MODEL = "yolo11n.pt"
model = None

def get_model():
    global model
    if model is None:
        model = YOLO(MODEL)
    return model


def detect_from_image(image_path: str, conf: float = 0.35, iou: float = 0.45):
    m = get_model()
    res = m.predict(image_path, conf=conf, iou=iou, verbose=False)[0]
    boxes = res.boxes
    out = []
    if boxes is not None:
        xyxy = boxes.xyxy.cpu().tolist()
        cls = boxes.cls.int().cpu().tolist()
        cf = boxes.conf.cpu().tolist()
        names = res.names
        for b, c, k in zip(xyxy, cls, cf):
            out.append({"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "class": names[int(c)], "conf": float(k)})
    return out


def detections_to_text(dets, top_k=10):
    c = Counter(d["class"] for d in dets)
    items = sorted(c.items(), key=lambda x: -x[1])[:top_k]
    if not items:
        return "no objects detected"
    parts = [f"{v} {k}{'s' if v != 1 and not k.endswith('s') else ''}" for k, v in items]
    return ", ".join(parts)


def summarize_text(dets_text: str, api_key: str = None, model_name: str = "gpt-4o-mini"):
    try:
        import openai
    except Exception:
        return "LLM unavailable (install openai). Summary placeholder: " + dets_text
    if not api_key:
        return "No API key provided. Raw detections: " + dets_text
    try:
        client = openai.OpenAI(api_key=api_key)
        system = "You are a concise security/retail scene analyst. Given object counts, write a short factual incident/situation summary in 2–4 lines. Avoid speculation."
        user = f"Scene objects: {dets_text}. Generate a neutral factual summary."
        resp = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.2, max_tokens=120)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}. Raw: {dets_text}"
