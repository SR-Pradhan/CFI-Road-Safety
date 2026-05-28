"""
VehicleTracker — wraps ByteTrack IDs from ultralytics to build unique vehicle records.

Deduplication: a vehicle seen across N frames is counted once.
Category mapping: COCO class → Indian MV Act classification.
"""

from collections import defaultdict


COCO_TO_CATEGORY = {
    "motorcycle": "2W",
    "bicycle":    "2W",
    "car":        "LMV",
    "truck":      "HMV",
    "bus":        "HMV",
}


class VehicleTracker:
    def __init__(self):
        # track_id → {"class", "category", "first_frame", "last_frame", "conf_max"}
        self.all_tracks: dict = {}

    def update(self, detections: list, frame_idx: int) -> list:
        """Update track records. Returns list of active tracked vehicles this frame."""
        active = []
        for det in detections:
            cls = det.get("class")
            if cls not in COCO_TO_CATEGORY:
                continue
            tid = det.get("track_id", -1)
            if tid == -1:
                continue

            if tid not in self.all_tracks:
                self.all_tracks[tid] = {
                    "track_id":   tid,
                    "class":      cls,
                    "category":   COCO_TO_CATEGORY[cls],
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "conf_max":   det["conf"],
                }
            else:
                self.all_tracks[tid]["last_frame"] = frame_idx
                self.all_tracks[tid]["conf_max"] = max(
                    self.all_tracks[tid]["conf_max"], det["conf"]
                )
            active.append(self.all_tracks[tid])
        return active
