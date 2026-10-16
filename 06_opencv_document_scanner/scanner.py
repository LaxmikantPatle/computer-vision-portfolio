import cv2
import numpy as np
from pathlib import Path


def order_points(pts):
    if pts is None or len(pts) == 0:
        return None
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    if rect is None:
        return image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def auto_scan(image_path: str, out_path: str = "scanned.jpg", debug: bool = False, blur_ksize: int = 5, canny_low: int = 75, canny_high: int = 200):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"could not read {image_path}")
    orig = img.copy()
    h, w = img.shape[:2]
    scale = 500.0 / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edged = cv2.Canny(blurred, canny_low, canny_high)
    if debug:
        cv2.imwrite("debug_edged.jpg", edged)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None:
        print("no 4-point contour found (fallback: whole image)")
        warped = orig
    else:
        screenCnt = screenCnt.reshape(4, 2) / scale
        warped = four_point_transform(orig, screenCnt)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_adap = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    cv2.imwrite(out_path, warped_adap)
    if debug:
        cv2.imwrite("debug_warped_color.jpg", warped)
    print(f"saved scanned document -> {out_path}")
    return out_path
