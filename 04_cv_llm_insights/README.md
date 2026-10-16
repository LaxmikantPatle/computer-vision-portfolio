# Project 4 - CV + LLM Image-to-Insight Pipeline

Convert YOLO detections on an image (or video frame) to a natural-language summary using an LLM API. This is a simple "smart surveillance summarizer" / retail scene analyzer demonstrating CV + LLM integration.

## Install

```bash
pip install ultralytics opencv-python numpy openai
```

## Usage

Set your API key (OpenAI compatible or OpenAI):
```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

Run on an image:

```bash
python run_insights.py --image test.jpg --api-key $env:OPENAI_API_KEY
```

Outputs `insights.json`:
```json
{
  "image": "test.jpg",
  "detections": [...],
  "counts_text": "3 persons, 1 car, 2 bikes",
  "summary": "The scene shows 3 people, 1 car, and 2 bicycles present."
}
```

## How it works

1. **YOLO detection** - runs YOLO11 on the image, returns boxes/classes/confidences.
2. **Text conversion** - aggregates by class (`Counter`) into a compact line: `"3 persons, 1 car"`.
3. **LLM summarization** - sends that text to an LLM with a constrained system prompt to produce a short, factual summary (or incident-style report).
4. **Return both** - preserves raw detections (for auditability) and the AI-generated summary.

## Notes on the blend (CV + LLMs + GenAI)

- **Minimal, explainable**: the LLM only sees object counts (not pixels), reducing cost/latency and avoiding hallucinating unseen details.
- **Swap providers**: you can replace `openai` with `requests` to call any OpenAI-compatible endpoint (Groq, Ollama, LMDeploy). Just adjust `summarize_text()` in `detect_to_insights.py`.
- **Video extension**: sample frames, run YOLO+LLM on keyframes, and stitch into a short report. (Easy to add later.)
- **Safety**: keep prompts factual; don't send PII in prompts.

## Interview talking points

- **Pipeline**: perception (CV/YOLO) -> symbolic representation (counts) -> reasoning/summarization (LLM).
- **Why counts first**: cheaper, faster, more grounded, easier to verify.
- **When to use multimodal (VLM)** vs text-only: VLM sees full context but costs more and is harder to control; text-only is a good pragmatic choice.
- **GenAI in CV**: turning raw detections into human-readable reports, alerts, or natural-language queries.
