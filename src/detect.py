import argparse
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import median

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Webcam fruit/coin counter")
    p.add_argument("--detector", choices=["hough", "yolo"], default="hough",
                   help="hough: classical circle finder, works for coins with no training. "
                        "yolo: neural detector (needs trained weights for coins).")
    p.add_argument("--weights", default="yolov8n.pt",
                   help="YOLO weights. Swap in your trained model later.")
    p.add_argument("--source", type=int, default=0, help="Webcam index")
    p.add_argument("--classes", nargs="*", default=["apple", "orange"],
                   help="YOLO class names to count. Pass empty (--classes) with a custom model.")
    p.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    p.add_argument("--min-r", type=int, default=20, help="Hough: min coin radius (px)")
    p.add_argument("--max-r", type=int, default=120, help="Hough: max coin radius (px)")
    p.add_argument("--sens", type=int, default=40,
                   help="Hough: accumulator threshold. Higher = fewer false circles, may miss coins")
    p.add_argument("--post", default="P001", help="Post ID attached to saved counts")
    p.add_argument("--out", default="counts.csv", help="CSV log file")
    return p.parse_args()


def make_hough(min_r: int, max_r: int, sens: int):
    def detect(frame):
        gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 5)
        # param2 = sens: lower finds more circles (and more false ones)
        c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 2,
                             param1=100, param2=sens, minRadius=min_r, maxRadius=max_r)
        circles = [] if c is None else c[0]
        out = frame.copy()
        for x, y, r in circles:
            cv2.circle(out, (int(x), int(y)), int(r), (0, 255, 0), 2)
        return len(circles), out
    return detect


def make_yolo(weights: str, classes: list[str], conf: float):
    from ultralytics import YOLO
    model = YOLO(weights)
    class_ids = None
    if classes:
        names = {v: k for k, v in model.names.items()}
        missing = [c for c in classes if c not in names]
        if missing:
            raise SystemExit(f"Unknown classes {missing}. Available: {list(names)}")
        class_ids = [names[c] for c in classes]

    def detect(frame):
        result = model.predict(frame, conf=conf, classes=class_ids, verbose=False)[0]
        return len(result.boxes), result.plot()
    return detect


def main():
    args = parse_args()
    detect = (make_hough(args.min_r, args.max_r, args.sens) if args.detector == "hough"
              else make_yolo(args.weights, args.classes, args.conf))

    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)  # DSHOW opens faster on Windows
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try --source 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    out_path = Path(args.out)
    new_file = not out_path.exists()
    history = deque(maxlen=15)  # rolling median smooths flicker
    post_id = args.post
    prev = time.time()

    print("Keys: [s] save count  [p] change post ID  [q] quit")

    with out_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "post_id", "count"])

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            n, annotated = detect(frame)
            history.append(n)
            smooth = int(median(history))

            fps = 1 / max(time.time() - prev, 1e-6)
            prev = time.time()
            cv2.putText(annotated, f"Post {post_id} | count: {smooth} | {fps:.0f} fps",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Fruit counter", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                writer.writerow([datetime.now().isoformat(timespec="seconds"),
                                 post_id, smooth])
                f.flush()
                print(f"Saved: post {post_id}, count {smooth}")
            elif key == ord("p"):
                post_id = input("New post ID: ").strip() or post_id

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
