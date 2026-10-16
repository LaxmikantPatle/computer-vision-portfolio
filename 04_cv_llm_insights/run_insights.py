import argparse, os, json
from detect_to_insights import detect_from_image, detections_to_text, summarize_text

def parse_args():
    p = argparse.ArgumentParser(description="Image -> YOLO detections -> LLM summary (incident report style)")
    p.add_argument("--image", required=True)
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--out", default="insights.json")
    return p.parse_args()

def main():
    args = parse_args()
    dets = detect_from_image(args.image, conf=args.conf)
    text = detections_to_text(dets)
    summary = summarize_text(text, api_key=args.api_key, model_name=args.model)
    result = {"image": args.image, "detections": dets, "counts_text": text, "summary": summary}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
