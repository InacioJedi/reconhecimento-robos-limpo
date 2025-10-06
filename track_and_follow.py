import argparse
import os
import sys
import cv2
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Detect and track robots in video or directory of videos with slow-motion and skip options.")
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pt)')
    parser.add_argument('--source', type=str, required=True, help='Path to video file or directory of video files')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help='Tracker config YAML')
    parser.add_argument('--output_dir', type=str, default='tracked_outputs', help='Directory for output videos')
    parser.add_argument('--delay', type=int, default=1, help='Playback delay in ms (higher = slower playback)')
    parser.add_argument('--skip', type=int, default=1, help='Process every Nth frame to reduce noisy detections')
    parser.add_argument('--augment', action='store_true', help='Enable augmentation during inferencing')
    return parser.parse_args()

def process_video(model, video_path, args):
    # Prepare results stream with augmentation if desired
    results = model.track(
        source=video_path,
        conf=args.conf,
        imgsz=args.imgsz,
        tracker=args.tracker,
        persist=True,
        stream=True,
        augment=args.augment
    )

    first = next(results, None)
    if first is None:
        print(f"Warning: No frames in file {video_path}")
        return

    frame = first.orig_img if hasattr(first, 'orig_img') and first.orig_img is not None else first.imgs[0]
    # Interactive ROI selection
    print(f"Processing {video_path}: select ROI. Drag and ENTER/SPACE. C to cancel.")
    roi = cv2.selectROI('Select ROI', frame, False)
    cv2.destroyWindow('Select ROI')
    x1, y1, w, h = roi
    x2, y2 = x1 + w, y1 + h

    # Setup writer with slowed fps
    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.basename(video_path)
    name, _ = os.path.splitext(basename)
    out_path = os.path.join(args.output_dir, f"tracked_{name}.mp4")
    orig_fps = first.metadata.get('fps', 30) if hasattr(first, 'metadata') else 30
    slow_fps = max(1, orig_fps // args.skip)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, slow_fps, (w, h))

    # Process frames with skipping
    count = 0
    for result in [first] + list(results):
        if count % args.skip != 0:
            count += 1
            continue
        frame = result.orig_img if hasattr(result, 'orig_img') and result.orig_img is not None else result.imgs[0]
        crop = frame[y1:y2, x1:x2]
        # Draw boxes
        for box in result.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            cx, cy = (bx1 + bx2)//2, (by1 + by2)//2
            if cx < x1 or cx > x2 or cy < y1 or cy > y2:
                continue
            ax1, ay1 = bx1-x1, by1-y1
            ax2, ay2 = bx2-x1, by2-y1
            cls_id = int(box.cls[0]); conf = float(box.conf[0])
            tid = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else -1
            label = f"ID:{tid} {model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(crop, (ax1, ay1), (ax2, ay2), (0,255,0), 2)
            cv2.putText(crop, label, (ax1, ay1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imshow('Tracked ROI', crop)
        if args.delay > 1:
            cv2.waitKey(args.delay)
        else:
            cv2.waitKey(1)
        out.write(crop)
        count += 1
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved slowed/skip-tracked video to {out_path}")

if __name__ == '__main__':
    args = parse_args()
    model = YOLO(args.weights)
    if os.path.isdir(args.source):
        vids = sorted([os.path.join(args.source,f) for f in os.listdir(args.source) if f.lower().endswith(('.mp4','.avi','.mkv','.mov'))])
        for v in vids:
            process_video(model, v, args)
    else:
        process_video(model, args.source, args)
