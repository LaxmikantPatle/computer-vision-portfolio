import cv2
import numpy as np

def make_test_document(path="test_document.jpg"):
    # Create a document with strong contrast border (dark desk, light paper)
    w, h = 1200, 900
    desk = np.full((h, w, 3), 30, dtype=np.uint8)  # dark background
    # document corners (slightly rotated rectangle)
    doc = np.full((400, 300, 3), 240, dtype=np.uint8)
    cv2.putText(doc, "INVOICE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20,20,20), 2)
    for i in range(1, 12):
        y = 80 + i*25
        cv2.line(doc, (20, y), (280, y), (180,180,180), 1)
        cv2.putText(doc, f"Line {i}", (20, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40,40,40), 1)
    # Place document with perspective transform into scene
    src = np.float32([[0,0],[300,0],[300,400],[0,400]])
    dst = np.float32([[200,150],[800,120],[820,700],[220,720]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(doc, M, (w, h))
    # Composite: where warped doc has content, overlay onto desk
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    inv = 255 - mask
    desk_bg = cv2.bitwise_and(desk, desk, mask=inv)
    doc_fg = cv2.bitwise_and(warped, warped, mask=mask)
    combined = cv2.add(desk_bg, doc_fg)
    cv2.imwrite(path, combined)
    print("wrote", path)

if __name__ == "__main__":
    make_test_document()
