"""
Task 4: Object Detection and Tracking
--------------------------------------
- Real-time video input via OpenCV (webcam or video file)
- Pre-trained YOLOv8 model for object detection
- Per-frame detection with bounding boxes
- Deep SORT for multi-object tracking (persistent IDs)
- Live display of labels + tracking IDs

Usage:
    python object_detection_tracking.py                # use default webcam (index 0)
    python object_detection_tracking.py --source video.mp4
    python object_detection_tracking.py --source 0 --model yolov8s.pt --conf 0.4
"""

import argparse
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + Deep SORT real-time object detection and tracking")
    parser.add_argument("--source", default=r"C:\Users\Muhammad Wahab\Documents\CodeAlphaIntenship\Project3\VOT-Basket.mp4", help="Video source: webcam index or path to video file")
    parser.add_argument("--model", default="yolov8n.pt", help="Path/name of pre-trained YOLO model")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--max-age", type=int, default=30, help="Deep SORT: frames to keep a lost track alive")
    return parser.parse_args()


def get_video_source(source_arg):
    # Allow numeric strings to be treated as webcam indices
    if source_arg.isdigit():
        return int(source_arg)
    return source_arg


def main():
    args = parse_args()

    # 1. Set up real-time video input (webcam or video file)
    source = get_video_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    # 2. Load a pre-trained detection model (YOLOv8)
    model = YOLO(args.model)
    class_names = model.names

    # 3. Initialize the tracker (Deep SORT)
    tracker = DeepSort(max_age=args.max_age)

    window_name = "Object Detection & Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Press 'q' or 'Esc' (with the video window focused) to quit. Or Ctrl+C in the terminal.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or camera error.")
                break

            # 4. Run detection on the current frame
            results = model(frame, conf=args.conf, verbose=False)[0]

            # Format detections for Deep SORT: [[x, y, w, h], confidence, class_id]
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls_id))

            # 5. Update tracker with this frame's detections
            tracks = tracker.update_tracks(detections, frame=frame)

            # Draw bounding boxes with class label + tracking ID
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                cls_id = track.get_det_class()
                label = class_names.get(cls_id, "object") if cls_id is not None else "object"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} ID:{track_id}",
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # Display output in real time
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or Esc
                print("Quit key pressed. Stopping...")
                break

            # Also stop if the user closes the window with the X button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed. Stopping...")
                break
    except KeyboardInterrupt:
        print("Ctrl+C pressed. Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # flush any pending GUI events so the window actually closes on Windows


if __name__ == "__main__":
    main()