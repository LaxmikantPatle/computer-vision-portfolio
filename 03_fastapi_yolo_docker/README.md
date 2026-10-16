# Project 3 - FastAPI YOLO Detection Endpoint + Docker

Minimal FastAPI service that accepts an image via `POST /detect` and returns JSON with bounding boxes, classes, and confidences. Includes a Dockerfile to containerize the service.

## Install locally

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --reload
# or: python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/docs` for Swagger UI and test `POST /detect`.

Example cURL:

```bash
curl -X POST "http://127.0.0.1:8000/detect?conf=0.3" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@test.jpg"
```

## Docker

Build and run:

```bash
docker build -t yolo-detect-api .
docker run -p 8000:8000 --rm yolo-detect-api
```

Test the container:

```bash
curl -X POST "http://localhost:8000/detect" -F "file=@test.jpg"
```

## Endpoint

- `GET /health` - health check
- `POST /detect` - multipart form with `file`. Query params: `conf` (0–1), `iou` (0–1), `imgsz` (int)

Response:
```json
{
  "count": 3,
  "inference_ms": 42.1,
  "detections": [
    {"x1": 10, "y1": 20, "x2": 100, "y2": 80, "class_id": 0, "class": "person", "confidence": 0.89}
  ]
}
```

## Notes

- Uses `opencv-python-headless` in requirements (no GUI libs needed in container).
- Model `yolo11n.pt` is downloaded by Ultralytics on first load; Dockerfile tries a warm cache on build.
- For production, consider volume mounting weights, model quantization, batching, GPU support, and auth.
