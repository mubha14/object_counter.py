# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:52:52 2026

@author: user
"""

"""
Task 5: Basic Object Counting Zone (Virtual Tripwire)
Barakah TechLabs - Computer Vision Automation Engineer Internship

Usage:
    python object_counter.py --source highway.mp4
    python object_counter.py --source highway.mp4 --line 0 400 1280 400 --save out.mp4
Press 'q' to quit.
"""

import argparse
from collections import defaultdict

import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def parse_args():
    p = argparse.ArgumentParser(description="Vehicle counting with virtual tripwire")
    p.add_argument("--source", default="highway.mp4", help="Path to highway video")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--line", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                   default=None, help="Tripwire endpoints (default: horizontal at 60%% height)")
    p.add_argument("--save", default=None, help="Optional output video path")
    return p.parse_args()


def side_of_line(point, p1, p2):
    return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])


def segments_intersect(a, b, c, d):
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    if args.line:
        p1, p2 = (args.line[0], args.line[1]), (args.line[2], args.line[3])
    else:
        y = int(height * 0.6)
        p1, p2 = (0, y), (width, y)

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))

    model = YOLO(args.model)

    prev_centroids = {}
    counted_ids = set()          # prevents double-counting
    total_count = 0
    class_counts = defaultdict(int)
    direction_counts = {"down/right": 0, "up/left": 0}
    trails = defaultdict(list)

    print("[INFO] Running... press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              classes=list(VEHICLE_CLASSES.keys()),
                              conf=args.conf, verbose=False)[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            ids = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), tid, cls in zip(boxes, ids, classes):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                centroid = (cx, cy)
                label = VEHICLE_CLASSES.get(cls, "vehicle")

                if tid in prev_centroids and tid not in counted_ids:
                    prev = prev_centroids[tid]
                    if segments_intersect(prev, centroid, p1, p2):
                        counted_ids.add(tid)
                        total_count += 1
                        class_counts[label] += 1
                        if side_of_line(centroid, p1, p2) > 0:
                            direction_counts["down/right"] += 1
                        else:
                            direction_counts["up/left"] += 1

                prev_centroids[tid] = centroid
                trails[tid].append(centroid)
                trails[tid] = trails[tid][-20:]

                color = (0, 255, 0) if tid in counted_ids else (255, 160, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, centroid, 4, (0, 0, 255), -1)
                cv2.putText(frame, f"{label} #{tid}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                for i in range(1, len(trails[tid])):
                    cv2.line(frame, trails[tid][i - 1], trails[tid][i], color, 1)

        cv2.line(frame, p1, p2, (0, 0, 255), 3)

        panel_h = 40 + 24 * (len(class_counts) + 2)
        cv2.rectangle(frame, (0, 0), (280, panel_h), (0, 0, 0), -1)
        cv2.putText(frame, f"TOTAL COUNT: {total_count}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        row = 55
        for name, n in class_counts.items():
            cv2.putText(frame, f"{name}: {n}", (10, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            row += 24
        for name, n in direction_counts.items():
            cv2.putText(frame, f"{name}: {n}", (10, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1, cv2.LINE_AA)
            row += 24

        if writer:
            writer.write(frame)

        cv2.imshow("Task 5 - Object Counting Zone", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n===== FINAL RESULT =====")
    print(f"Total vehicles counted: {total_count}")
    for name, n in class_counts.items():
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()