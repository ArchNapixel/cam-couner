# CLAUDE.md: Smart Farming Crop Counter

This file is the project brief for Claude Code. Read it fully before doing anything, then follow the "First session tasks" section (skip any task already done; check the repo first).

## 1. Project summary

A software-first smart farming tool that helps farmers monitor crops. The first feature is a **fruit counter**: detect and count fruits from camera images, attach counts to a plot position (post/row/zone), and show the farmer where yield is concentrated.

- **Eventual target crop:** dragon fruit (grown on posts/trellises, fruits hang around the top of each post).
- **Current stand-in subject:** **coins**. They let us build and validate the full pipeline (collect, label, train, evaluate, deploy) with free, instantly available data before we have dragon fruit photos.
- The inspiration is a thesis project that loads a map of a plot through a camera and counts the current number of dragon fruits.

## 2. Owner context and constraints

- The developer is **experienced**. Skip beginner explanations; be concise and direct. Explain the *why* of non-obvious design decisions briefly.
- **Budget: keep cost at the minimum.** Prefer free tools: open-source libraries, free Colab GPU, free-tier annotation tools, Open-Meteo style free APIs. Do not introduce paid services or dependencies without asking first.
- **Prefers software.** Hardware (ESP32, sensors) is optional and only if clearly worth it. AI/computer vision is in scope and is the core of the project.
- Development environment: **VS Code** on a local machine (Windows), Python virtual environment, **CPU only locally**. Training happens in **Google Colab** (free GPU). Webcam (or a phone used as a webcam via DroidCam/Iriun) is the current camera.

## 3. What has been decided so far

1. **Counting unit = per post (v1).** The farmer photographs each post from a few angles; the app counts fruits per post and tags a post ID. This avoids cross-image double counting. Video walk-through with a multi-object tracker (ByteTrack/BoT-SORT) is a later capture mode. GPS is used only to assign photos to zones, never to match individual fruits (phone GPS is accurate to only a few meters).
2. **Model:** fine-tune a small YOLO detector (nano/small), single class first (`coin`, later `fruit`, then ripe/unripe). Start from pretrained weights.
3. **Licensing note:** Ultralytics YOLO is AGPL-3.0. Fine for a thesis/personal project. If this is ever commercialized, evaluate Apache-licensed alternatives (RT-DETR, YOLOX) or the Ultralytics enterprise license. Keep model code behind a thin interface so the detector can be swapped.
4. **Success metric = count error, not mAP.** Compare predicted count vs. manual count per image/post. Report mean absolute error (MAE) and mean percent error. mAP is secondary.
5. **Coins first.** Coins prove the pipeline, not the field accuracy. They lack occlusion, camouflage, outdoor lighting, and scale variation, so never present coin results as evidence of dragon fruit accuracy.

## 4. Target architecture (build incrementally)

1. **Capture:** PWA or React Native app that takes photos, records GPS, asks for post/row ID. Offline-first.
2. **Inference:** start as local scripts, then a small FastAPI service. Later export to ONNX/TFLite for on-device counting with zero server cost.
3. **Storage:** SQLite first (Postgres + PostGIS only if needed). Record: post ID, timestamp, count, class breakdown, photo path, optional lat/lon.
4. **Map and dashboard:** Leaflet or MapLibre with OpenStreetMap tiles. Plot drawn as a GeoJSON polygon divided into row/post zones, colored by count (yield heatmap).

Do not build steps 1, 3, or 4 until the detector pipeline (step 2) meets the success criteria below.

## 5. Repo layout

```
cam-couner/
  CLAUDE.md
  README.md
  requirements.txt
  .gitignore
  data/
    raw/            # original photos (gitignored)
    coins/          # YOLO-format dataset: images/{train,val,test}, labels/{train,val,test}, data.yaml
    truth_counts.csv  # filename,true_count  (ground truth for evaluation)
  models/           # trained weights, e.g. best.pt (gitignored)
  runs/             # training/inference outputs (gitignored)
  src/
    detect.py       # live webcam detection + counting + CSV logging
    count.py        # counting logic reusable by other modules
    evaluate.py     # count-error evaluation against truth_counts.csv
    api.py          # FastAPI service (later)
  notebooks/
    train_colab.ipynb
  tests/
```

## 6. Conventions

- Python 3.10+. Type hints on public functions. Format/lint with `ruff` (add to dev requirements).
- Keep scripts runnable from the repo root: `python src/detect.py ...`.
- Use `argparse` for CLI options with sensible defaults. No hard-coded absolute paths.
- Keep the detector behind a small wrapper (`load_model`, `predict(frame) -> list[Detection]`) so the model can be swapped.
- Large files (images, weights, runs) stay out of git. Maintain `.gitignore`. Mention `git-lfs` only if the developer asks.
- Prefer small, reviewable changes. Run the code you write when possible, and report actual results rather than assumptions.
- Ask before: adding paid services, adding heavy dependencies, restructuring the repo, or changing decisions in section 3.
- Be honest about uncertainty. If a result looks too good (for example high accuracy on a tiny, easy test set), say so and suggest a harder test.

## 7. Current state

- Repo was empty at the time this file was written (only an initial commit). Nothing below has been scaffolded or verified yet.
- Environment setup and a starter webcam script have been drafted (see below) but **not yet verified in this repo**.
- No dataset collected yet. No model trained yet.
- The starter script performs **inference only** with pretrained `yolov8n.pt` (COCO). It does not train anything. COCO has no coin or dragon fruit class, so for the pre-training smoke test it counts `apple`/`orange` as stand-ins.

### Starter `requirements.txt`

```
ultralytics
opencv-python
```

(Add `pandas`, `numpy`, `matplotlib`, and `ruff` as needed for evaluation and dev.)

### Starter `src/detect.py`

```python
import argparse
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import median

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Webcam fruit counter")
    p.add_argument("--weights", default="yolov8n.pt",
                   help="Path to weights. Swap in your trained model later.")
    p.add_argument("--source", type=int, default=0, help="Webcam index")
    p.add_argument("--classes", nargs="*", default=["apple", "orange"],
                   help="Class names to count. Pass empty (--classes) with a custom model.")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--post", default="P001", help="Post ID attached to saved counts")
    p.add_argument("--out", default="counts.csv", help="CSV log file")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    class_ids = None
    if args.classes:
        names = {v: k for k, v in model.names.items()}
        missing = [c for c in args.classes if c not in names]
        if missing:
            raise SystemExit(f"Unknown classes {missing}. Available: {list(names)}")
        class_ids = [names[c] for c in args.classes]

    cap = cv2.VideoCapture(args.source)
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

            result = model.predict(frame, conf=args.conf, classes=class_ids,
                                   verbose=False)[0]
            history.append(len(result.boxes))
            smooth = int(median(history))

            annotated = result.plot()
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
```

Improve this as needed (for example move logic into `count.py`, add a detector wrapper), but keep the behavior.

## 8. Roadmap

### Phase 0: Environment and smoke test
- Create `.venv`, install requirements, set up `.gitignore`, `README.md`, folder layout.
- Run `detect.py` with the pretrained model on a webcam and confirm it works (apple/orange stand-ins or photos on a screen).

### Phase 1: Coin dataset
- Developer takes ~100 to 150 photos: coins on varied surfaces (table, cloth, paper), 1 to 20 coins per shot, varied spacing, some touching/overlapping, varied angles and lighting. Keep resolution and camera distance reasonably consistent.
- Record the **true count per photo** in `data/truth_counts.csv`.
- Label with Roboflow (free tier) or Label Studio (`pip install label-studio`), single class `coin`. Pre-label with a model and correct by hand to save time.
- Split roughly 70/20/10 or 80/20 train/val, plus a **held-out test set that is never used for training or tuning**. Export in YOLO format with a `data.yaml`.

### Phase 2: Train in Colab
- Write `notebooks/train_colab.ipynb`: mount Drive or upload the dataset, install `ultralytics`, then fine-tune:
  ```python
  from ultralytics import YOLO
  model = YOLO("yolov8n.pt")
  model.train(data="data.yaml", epochs=50, imgsz=640)
  ```
- Download `runs/detect/train/weights/best.pt` into `models/`.

### Phase 3: Evaluate counts
- Write `src/evaluate.py`: run the model over the held-out images, compare predicted vs. true counts from `truth_counts.csv`, report MAE, mean percent error, and a per-image table. Plot predicted vs. true.
- Success target for the coin pipeline: MAE under ~1 coin on easy shots. **Viability bar for the eventual dragon fruit model: roughly 15 to 20% count error.**

### Phase 4: Stress test
- Deliberately make it harder (stacked/partial coins, busy tablecloth, dim light) and document where error grows. This previews occlusion and lighting problems in the field.

### Phase 5: Swap to dragon fruit
- Collect 150 to 200+ real photos (mixed time of day, distance, ripeness). Prefer a **phone camera** and keep capture conditions consistent. Real field data is the biggest factor in success.
- Single class `fruit` first, then ripe/unripe, then flower/young fruit for harvest forecasting.
- Fine-tune again, evaluate with the same count-error metric against manual counts per post.

### Phase 6: Product layer (only after the detector is viable)
- Per-post pipeline with post IDs and SQLite storage, then a FastAPI service, then a PWA/React Native capture app, then the map heatmap with Leaflet/MapLibre, then optional video mode with tracking.

## 9. Known technical challenges (keep in mind)

- **Occlusion:** hidden fruits will be undercounted. Use multiple angles per post and calibrate with a correction factor measured against manual counts.
- **Camouflage:** green unripe fruit on green stems is the hardest class.
- **Lighting:** harsh midday sun and shadows hurt accuracy. Capture guidelines (time of day, distance) are part of the product.
- **Camera domain shift:** a model trained on webcam images may not transfer to phone photos. Train on the same kind of camera that will be used in deployment.
- **Double counting** only becomes an issue in video/overlapping-photo modes. The per-post design avoids it in v1.

## 10. First session tasks

1. Read this file, then inspect the repo (it may be empty or partially set up).
2. Scaffold the repo layout from section 5, create `.gitignore`, `README.md`, `requirements.txt`, and a `.venv` setup note.
3. Create `src/detect.py` from section 7 and verify it runs against the webcam. Report any camera index issues.
4. Draft `src/evaluate.py` and a sample `data/truth_counts.csv` format so it is ready when photos exist.
5. Draft `notebooks/train_colab.ipynb` for the coin dataset.
6. Summarize what was done, what was verified vs. assumed, and the next manual step for the developer (collecting and labeling coin photos).

Do not start Phase 6 work. Ask before deviating from any decision in section 3.
