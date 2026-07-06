# Task 4: Object Detection and Tracking

Real-time object detection (YOLOv8) + multi-object tracking (Deep SORT) using a webcam or video file.

## How it maps to the task checklist

| Checklist item | Implementation |
|---|---|
| Real-time video input | `cv2.VideoCapture` on webcam index or video file path |
| Pre-trained model | `ultralytics.YOLO("yolov8n.pt")` — auto-downloads on first run |
| Process each frame, draw boxes | `model(frame)` per loop iteration, boxes drawn with `cv2.rectangle` |
| Object tracking (SORT / Deep SORT) | `deep_sort_realtime.DeepSort` assigns persistent IDs frame-to-frame |
| Display output with labels + IDs | `cv2.imshow` window shows class name + track ID above each box |

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# Default webcam
python object_detection_tracking.py

# Specific webcam index
python object_detection_tracking.py --source 1

# Video file
python object_detection_tracking.py --source path/to/video.mp4

# Larger/more accurate model, custom confidence threshold
python object_detection_tracking.py --model yolov8s.pt --conf 0.5
```

Press **q** or **Esc** (with the video window focused) to quit. You can also close the window with the **X** button, or press **Ctrl+C** in the terminal — all four are handled cleanly.

## Notes

- Model options (smallest/fastest → largest/most accurate): `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`. The `.pt` weights download automatically the first time you run with a given model name.
- `--max-age` controls how many frames Deep SORT keeps a track alive without a matching detection (useful if objects are briefly occluded).
- To restrict detection to specific classes (e.g. only people and cars), you can filter `results.boxes` by `cls_id` before passing detections to the tracker — ask if you'd like this added.
- If detections flicker across nearby classes, raising `--conf` (e.g. to 0.5–0.6) usually helps.

## Troubleshooting

**Pressing `q` doesn't stop it (Windows):**
- Make sure you click the video window's title bar first so it has keyboard focus — clicking the terminal instead means the keypress never reaches OpenCV.
- Check for a conflicting headless OpenCV install, which can break window/keyboard handling:
  ```bash
  python -m pip show opencv-python-headless
  pip uninstall opencv-python-headless
  ```
- As a guaranteed fallback, click the terminal and press `Ctrl+C`.

**File paths with spaces (e.g. `.............`):**
Wrap the path in quotes and use forward slashes to avoid Windows backslash-escape errors:
```bash
python object_detection_tracking.py --source "C:/Users/Muhammad Wahab/Documents/CodeAlphaIntenship/Project3/VOT-Woman.mp4"
```
Do not hardcode a Windows path into the script's `default=` value using plain backslashes — either use `--source` on the command line, or prefix a hardcoded path with `r` (e.g. `default=r"C:\path\to\video.mp4"`).

