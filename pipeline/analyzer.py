"""
VideoAnalyzer — orchestrates per-frame detection, tracking, and violation logic.

Model choices:
  - Object detection : YOLOv8x (COCO) via ultralytics
      Covers: person, motorcycle, car, bus, truck, bicycle, traffic_light, cell_phone
  - Helmet detection : YOLOv8 fine-tuned on Indian road data
      Weights: models/helmet_yolov8.pt  (see README for download link)
      Fallback: heuristic head-region analysis when weights absent
  - Tracking        : ByteTrack (built into ultralytics .track())
"""

import cv2
import numpy as np
import time
from collections import defaultdict
from pathlib import Path

from pipeline.violations import ViolationDetector
from pipeline.tracker import VehicleTracker
from pipeline.junction import JunctionDetector
from pipeline.vehicles import VehicleClassifier


# COCO class IDs relevant to this task
COCO = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "traffic_light": 9,
    "cell_phone": 67,
}


class VideoAnalyzer:
    def __init__(self, conf_threshold=0.45, device="cpu", frame_skip=2):
        self.conf = conf_threshold
        self.device = device
        self.frame_skip = frame_skip
        self._load_models()

    def _load_models(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8x.pt")  # auto-downloads on first run
            self.model.to(self.device)
            print(f"[✓] YOLOv8x loaded on {self.device}")
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

        # Helmet model — optional, graceful fallback
        helmet_path = Path("models/helmet_yolov8.pt")
        if helmet_path.exists():
            from ultralytics import YOLO as _YOLO
            self.helmet_model = _YOLO(str(helmet_path))
            self.helmet_model.to(self.device)
            print("[✓] Helmet model loaded")
        else:
            self.helmet_model = None
            print("[!] Helmet model not found — using heuristic fallback (lower accuracy)")

        self.violation_detector = ViolationDetector(self.helmet_model)
        self.tracker = VehicleTracker()
        self.junction_detector = JunctionDetector()
        self.vehicle_classifier = VehicleClassifier()

    def analyze(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[•] Video: {Path(video_path).name} | {width}×{height} @ {fps:.1f}fps | {total_frames} frames")

        # ── Storage ────────────────────────────────────────────────────────────
        violation_events = defaultdict(list)   # violation_type → [event, ...]
        junction_events  = []
        vehicle_density  = []                  # (timestamp, count)
        frame_annotations = defaultdict(list)  # frame_idx → [annotation, ...]

        frame_idx = 0
        prev_gray = None
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % self.frame_skip != 0:
                continue

            timestamp = frame_idx / fps
            ts_str = self._fmt_ts(timestamp)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── YOLO detection ─────────────────────────────────────────────────
            results = self.model.track(
                frame,
                conf=self.conf,
                persist=True,          # ByteTrack across frames
                verbose=False,
                classes=list(COCO.values()),
            )

            detections = self._parse_detections(results, frame.shape)

            # ── Vehicle tracking & classification ──────────────────────────────
            tracked_vehicles = self.tracker.update(detections, frame_idx)
            vehicle_density.append({
                "timestamp": round(timestamp, 2),
                "ts_str": ts_str,
                "count": len([d for d in detections if d["class"] in ("car","motorcycle","bus","truck","bicycle")]),
            })

            # ── Violation detection ────────────────────────────────────────────
            frame_violations = self.violation_detector.check(
                frame, detections, prev_gray, gray, fps, timestamp, ts_str
            )
            for v in frame_violations:
                violation_events[v["type"]].append(v)
                if len(frame_annotations[frame_idx]) < 10:
                    frame_annotations[frame_idx].append(v)

            # ── Junction detection ─────────────────────────────────────────────
            junction = self.junction_detector.detect(frame, gray, prev_gray, timestamp, ts_str)
            if junction:
                junction_events.append(junction)

            prev_gray = gray

            # Progress
            if frame_idx % 300 == 0:
                elapsed = time.time() - start_time
                pct = frame_idx / max(total_frames, 1) * 100
                print(f"  [{pct:.0f}%] frame {frame_idx}/{total_frames} | {elapsed:.1f}s elapsed")

        cap.release()

        # ── Compile final results ─────────────────────────────────────────────
        vehicle_summary = self.vehicle_classifier.summarize(self.tracker.all_tracks)
        violation_summary = self._compile_violations(violation_events)
        junction_summary = self._compile_junctions(junction_events)

        return {
            "meta": {
                "video": Path(video_path).name,
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_seconds": round(total_frames / fps, 1),
                "processed_frames": frame_idx,
                "device": self.device,
            },
            "violations": violation_summary,
            "junctions": junction_summary,
            "vehicles": vehicle_summary,
            "density_timeline": vehicle_density,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _parse_detections(self, results, frame_shape) -> list:
        """Convert ultralytics Results to a flat list of detection dicts."""
        inv_coco = {v: k for k, v in COCO.items()}
        detections = []
        h, w = frame_shape[:2]
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in inv_coco:
                    continue
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                track_id = int(boxes.id[i].item()) if boxes.id is not None else -1
                detections.append({
                    "class": inv_coco[cls_id],
                    "class_id": cls_id,
                    "conf": round(conf, 3),
                    "bbox": xyxy.tolist(),   # [x1,y1,x2,y2]
                    "track_id": track_id,
                })
        return detections

    def _compile_violations(self, events: dict) -> dict:
        VIOLATION_LABELS = {
            "helmet_less":   "Helmet-less Riding",
            "wrong_side":    "Wrong-side Driving",
            "signal_jump":   "Signal Jumping",
            "phone_use":     "Mobile Phone Use",
            "triple_riding": "Triple Riding",
        }
        summary = {}
        for key, label in VIOLATION_LABELS.items():
            evs = events.get(key, [])
            # top-3 by confidence for bounding box annotations
            top3 = sorted(evs, key=lambda e: e.get("conf", 0), reverse=True)[:3]
            summary[key] = {
                "label": label,
                "total_count": len(evs),
                "instances": [
                    {"ts": e["ts"], "frame": e["frame"], "conf": e.get("conf", 0)}
                    for e in evs
                ],
                "top3_annotations": [
                    {"ts": e["ts"], "frame": e["frame"], "bbox": e.get("bbox"), "conf": e.get("conf", 0)}
                    for e in top3
                ],
            }
        return summary

    def _compile_junctions(self, events: list) -> dict:
        type_counts = defaultdict(int)
        for e in events:
            type_counts[e["junction_type"]] += 1
        return {
            "total_count": len(events),
            "type_breakdown": dict(type_counts),
            "instances": events,
        }

    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
