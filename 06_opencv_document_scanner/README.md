# Project 6 - Classical OpenCV Document Scanner

A pure classical-CV document scanner (no deep learning) that finds a document in an image, corrects its perspective, and enhances it for OCR/reading. Built with core OpenCV building blocks: Gaussian blur, Canny edge detection, contour finding, `approxPolyDP`, `getPerspectiveTransform`, `warpPerspective`, and adaptive thresholding.

## Pipeline

1. **Resize** the image for faster processing (keep scale factor to map corners back to original).
2. **Grayscale + Gaussian blur** to reduce noise.
3. **Canny edge detection** to highlight strong edges (the document boundary).
4. **Contour extraction** (`findContours` + sort by area) to find the largest candidate region.
5. **Polygon approximation** (`approxPolyDP`) with epsilon ~2% of perimeter; keep the first 4-point contour.
6. **Order points** (top-left, top-right, bottom-right, bottom-left).
7. **Perspective transform** (`getPerspectiveTransform` + `warpPerspective`) to a flat rectangle sized by the max edge lengths.
8. **Adaptive threshold** for a clean, OCR-friendly binary output.

## Install

```bash
pip install opencv-python numpy
```

## Run

```bash
# Scan a photo of a document
python main.py --image photo_of_doc.jpg --out scanned.jpg

# Debug: save intermediate Canny edges and color-warped version
python main.py --image photo_of_doc.jpg --out scanned.jpg --debug

# Headless (no GUI window)
python main.py --image photo_of_doc.jpg --out scanned.jpg --no-show
```

Generate a test image (synthetic tilted document on a desk):

```bash
python make_test_doc.py
python main.py --image test_document.jpg --out scanned.jpg --debug --no-show
```

Outputs:
- `scanned.jpg` - flat, binarized document ready for OCR (Tesseract, etc.)
- `debug_edged.jpg` - Canny edges (when `--debug`)
- `debug_warped_color.jpg` - color perspective-corrected page (when `--debug`)

## Why this works (interview talking points)

- **Canny edges** mark strong intensity gradients — the paper/background boundary.
- **Contours** are connected edge regions; sorting by area picks the biggest likely page.
- **`approxPolyDP`** simplifies the contour to a polygon; epsilon controls fidelity. For a quadrilateral page we look for 4 vertices.
- **Perspective transform** maps the 4 detected corners to a rectangle, removing camera tilt — exactly what you need before OCR.
- **Adaptive threshold** handles uneven lighting on paper better than a global threshold.

## Extending

- **Lane detection variant**: same idea with Hough lines on Canny edges (bonus mini-project).
- **Auto-crop** with slight margin to remove desk edges.
- **OCR hook**: pipe `scanned.jpg` into `pytesseract` or Google Vision for text extraction.
- **Webcam mode**: capture a frame, scan it live (similar to `main.py` in Project 1).

## Interview talking points

- **Edge vs contour**: edges are pixel-level gradients; contours are geometric boundaries derived from edges.
- **Why not just Hough lines?** Contours give a closed region directly, which is simpler for document segmentation than assembling line segments into a quadrilateral.
- **Failure cases**: shadows, low contrast paper, multi-page stacks, strong background texture. Mitigations: lighting normalization, more robust quadrilateral scoring (e.g., prefer convex, near-rectangular aspect), or ML-based detection (Project 1's YOLO) to localize the document first.
