import argparse, cv2
from scanner import auto_scan

def parse_args():
    p = argparse.ArgumentParser(description="Document scanner (edge detection + contour + perspective transform)")
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="scanned.jpg")
    p.add_argument("--debug", action="store_true", help="save debug images (edges, warped color)")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    auto_scan(args.image, out_path=args.out, debug=args.debug)
    if not args.no_show:
        scan = cv2.imread(args.out)
        if scan is not None:
            cv2.imshow("scanned", scan); cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
