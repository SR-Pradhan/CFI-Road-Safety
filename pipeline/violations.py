"""
ViolationDetector — rule-based logic applied on top of YOLO detections.

Each check method returns a list of violation event dicts.

Limitations documented honestly:
  - helmet_less  : best with fine-tuned model; heuristic used as fallback
  - wrong_side   : optical-flow heuristic; works well on clear roads
  - signal_jump  : requires visible traffic light + stop line in frame
  - phone_use    : requires cell_phone class detected near driver region
  - triple_riding: person-count overlap on motorcycle bounding box
"""

import cv2
import numpy as np
from typing import Optional


# Minimum IoU for a person bbox to be considered "on" a motorcycle
PERSON_MOTO_IOU_THRESH = 0.15

# Optical-flow threshold for wrong-side detection (px/frame)
WRONG_SIDE_FLOW_THRESH = 3.5

# Stop line is approximated at this fraction from bottom of frame
STOP_LINE_Y_FRACTION = 0.55


class ViolationDetector:
    def __init__(self, helmet_model=None):
        self.helmet_model = helmet_model
        # Cooldown: avoid duplicate events for same track within N seconds
        self._cooldown: dict = {}   # (violation_type, track_id) → last_ts
        self._cooldown_sec = 3.0

        # Track traffic-light state across frames
        self._tl_state: str = "unknown"  # red | green | yellow | unknown
        self._tl_cooldown = 0.0

    def check(self, frame, detections, prev_gray, curr_gray, fps, timestamp, ts_str) -> list:
        events = []
        motorbikes = [d for d in detections if d["class"] == "motorcycle"]
        persons    = [d for d in detections if d["class"] == "person"]
        phones     = [d for d in detections if d["class"] == "cell_phone"]
        tl_dets    = [d for d in detections if d["class"] == "traffic_light"]
        vehicles   = [d for d in detections
                      if d["class"] in ("car","motorcycle","bus","truck","bicycle")]

        # Update traffic light state
        if tl_dets:
            self._tl_state = self._classify_tl_color(frame, tl_dets[0]["bbox"])

        # Run each check
        events += self._check_helmet_less(frame, motorbikes, persons, timestamp, ts_str)
        events += self._check_triple_riding(motorbikes, persons, timestamp, ts_str)
        events += self._check_phone_use(frame, persons, phones, detections, timestamp, ts_str)
        if prev_gray is not None and curr_gray is not None:
            events += self._check_wrong_side(frame, vehicles, prev_gray, curr_gray, timestamp, ts_str, fps)
        events += self._check_signal_jump(frame, vehicles, timestamp, ts_str)

        return events

    # ── 1. Helmet-less riding ────────────────────────────────────────────────

    def _check_helmet_less(self, frame, motorbikes, persons, timestamp, ts_str) -> list:
        events = []
        for moto in motorbikes:
            riders = self._persons_on_vehicle(moto, persons)
            for rider in riders:
                key = ("helmet_less", moto["track_id"])
                if self._in_cooldown(key, timestamp):
                    continue

                has_helmet = self._has_helmet(frame, rider)
                if not has_helmet:
                    events.append(self._mk_event(
                        "helmet_less", timestamp, ts_str,
                        rider["bbox"], conf=rider["conf"]
                    ))
                    self._set_cooldown(key, timestamp)
        return events

    def _has_helmet(self, frame, rider_det) -> bool:
        """
        Two-stage check:
          1. If dedicated helmet model present — use it on the head ROI.
          2. Fallback: check if the top ~30% of the rider's bounding box
             contains a roughly circular / oval dark region (helmet silhouette).
        """
        x1, y1, x2, y2 = rider_det["bbox"]
        head_h = max(1, int((y2 - y1) * 0.30))
        head_roi = frame[y1:y1 + head_h, x1:x2]
        if head_roi.size == 0:
            return True  # can't determine → assume compliant (avoid false positive)

        if self.helmet_model is not None:
            try:
                res = self.helmet_model(head_roi, verbose=False)
                for r in res:
                    for cls in r.boxes.cls.tolist():
                        # class 0 = helmet in most fine-tuned helmet models
                        if int(cls) == 0:
                            return True
                return False
            except Exception:
                pass  # fall through to heuristic

        # Heuristic: large dark/coloured blob in head region → likely helmet
        gray_head = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_head, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        roi_area = head_roi.shape[0] * head_roi.shape[1]
        # If ≥20% of head ROI is covered by a dark blob → assume helmet present
        return area / max(roi_area, 1) >= 0.20

    # ── 2. Triple riding ─────────────────────────────────────────────────────

    def _check_triple_riding(self, motorbikes, persons, timestamp, ts_str) -> list:
        events = []
        for moto in motorbikes:
            riders = self._persons_on_vehicle(moto, persons)
            if len(riders) >= 3:
                key = ("triple_riding", moto["track_id"])
                if not self._in_cooldown(key, timestamp):
                    events.append(self._mk_event(
                        "triple_riding", timestamp, ts_str,
                        moto["bbox"], conf=moto["conf"],
                        meta={"rider_count": len(riders)}
                    ))
                    self._set_cooldown(key, timestamp)
        return events

    # ── 3. Phone use while driving ───────────────────────────────────────────

    def _check_phone_use(self, frame, persons, phones, all_dets, timestamp, ts_str) -> list:
        """
        Flags if a phone is detected with high overlap to a person who is also
        associated with a vehicle (i.e., likely the driver).
        """
        events = []
        if not phones:
            return events

        frame_h, frame_w = frame.shape[:2]
        vehicles = [d for d in all_dets if d["class"] in ("car","motorcycle","bus","truck")]

        for phone in phones:
            # Phone must be in the lower-center of frame (driver zone)
            px1, py1, px2, py2 = phone["bbox"]
            phone_cx = (px1 + px2) / 2
            phone_cy = (py1 + py2) / 2
            if phone_cy < frame_h * 0.3:
                continue  # Too high — likely not a driver

            # Find nearest person
            best_person = None
            best_iou = 0.0
            for p in persons:
                iou = self._iou(phone["bbox"], p["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_person = p

            # Also accept phone overlapping with a vehicle
            if best_iou < 0.01:
                for v in vehicles:
                    iou = self._iou(phone["bbox"], v["bbox"])
                    if iou > 0.05:
                        best_iou = iou
                        break

            if best_iou >= 0.01:
                key = ("phone_use", phone["track_id"])
                if not self._in_cooldown(key, timestamp):
                    bbox = best_person["bbox"] if best_person else phone["bbox"]
                    events.append(self._mk_event(
                        "phone_use", timestamp, ts_str,
                        bbox, conf=phone["conf"]
                    ))
                    self._set_cooldown(key, timestamp)
        return events

    # ── 4. Wrong-side driving ────────────────────────────────────────────────

    def _check_wrong_side(self, frame, vehicles, prev_gray, curr_gray, timestamp, ts_str, fps) -> list:
        """
        Uses sparse Lucas-Kanade optical flow on vehicle keypoints.
        India = left-hand traffic. Ego vehicle moves forward.
        Other vehicles should move rearward in frame (downward in optical flow).
        A vehicle moving strongly *upward* AND appearing in the ego lane is flagged.

        Limitations:
          - Requires clear road, consistent lighting
          - May miss subtle wrong-side incidents on complex intersections
          - Accuracy improves significantly with lane-detection model
        """
        events = []
        h, w = curr_gray.shape

        # Sample feature points from vehicle bounding boxes
        for veh in vehicles:
            x1, y1, x2, y2 = veh["bbox"]
            roi = prev_gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            pts = cv2.goodFeaturesToTrack(roi, maxCorners=20, qualityLevel=0.3, minDistance=7)
            if pts is None or len(pts) < 4:
                continue

            # Adjust coordinates to full frame
            pts_full = pts.copy()
            pts_full[:, :, 0] += x1
            pts_full[:, :, 1] += y1

            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts_full.astype(np.float32), None,
                winSize=(15, 15), maxLevel=2
            )
            if new_pts is None:
                continue

            good_old = pts_full[status == 1]
            good_new = new_pts[status == 1]
            if len(good_old) < 3:
                continue

            dy = np.mean(good_new[:, 0, 1] - good_old[:, 0, 1])   # positive = moving down in frame
            dx = np.mean(good_new[:, 0, 0] - good_old[:, 0, 0])

            # A vehicle moving strongly *upward* (dy < -threshold) is approaching
            # Check it's in the ego lane (centre 50% of frame width)
            cx = (x1 + x2) / 2
            in_ego_lane = w * 0.25 < cx < w * 0.75
            if dy < -WRONG_SIDE_FLOW_THRESH and in_ego_lane:
                key = ("wrong_side", veh["track_id"])
                if not self._in_cooldown(key, timestamp):
                    events.append(self._mk_event(
                        "wrong_side", timestamp, ts_str,
                        veh["bbox"], conf=min(0.9, abs(dy) / 10),
                        meta={"flow_dy": round(float(dy), 2)}
                    ))
                    self._set_cooldown(key, timestamp)
        return events

    # ── 5. Signal jumping ────────────────────────────────────────────────────

    def _check_signal_jump(self, frame, vehicles, timestamp, ts_str) -> list:
        """
        Detects vehicle crossing the virtual stop line while traffic light is red.
        Stop line is approximated at STOP_LINE_Y_FRACTION of frame height.
        Accuracy improves significantly when traffic light is clearly visible.
        """
        events = []
        if self._tl_state != "red":
            return events

        frame_h = frame.shape[0]
        stop_y = int(frame_h * STOP_LINE_Y_FRACTION)

        for veh in vehicles:
            x1, y1, x2, y2 = veh["bbox"]
            veh_bottom = y2
            # Vehicle bottom (front bumper in dashcam view) crossing stop line
            if veh_bottom > stop_y:
                key = ("signal_jump", veh["track_id"])
                if not self._in_cooldown(key, timestamp):
                    events.append(self._mk_event(
                        "signal_jump", timestamp, ts_str,
                        veh["bbox"], conf=0.75
                    ))
                    self._set_cooldown(key, timestamp)
        return events

    # ── Traffic light color classification ───────────────────────────────────

    def _classify_tl_color(self, frame, bbox) -> str:
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red: hue 0-10 or 160-180
        red_lo1 = cv2.inRange(hsv, np.array([0, 100, 100]),   np.array([10, 255, 255]))
        red_lo2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_lo1, red_lo2)

        green_mask  = cv2.inRange(hsv, np.array([40, 50, 50]),  np.array([90, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))

        counts = {
            "red":    cv2.countNonZero(red_mask),
            "green":  cv2.countNonZero(green_mask),
            "yellow": cv2.countNonZero(yellow_mask),
        }
        dominant = max(counts, key=counts.get)
        return dominant if counts[dominant] > 50 else "unknown"

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _persons_on_vehicle(self, vehicle, persons) -> list:
        on_vehicle = []
        for p in persons:
            if self._iou(vehicle["bbox"], p["bbox"]) >= PERSON_MOTO_IOU_THRESH:
                on_vehicle.append(p)
        return on_vehicle

    @staticmethod
    def _iou(boxA, boxB) -> float:
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / max(union, 1)

    @staticmethod
    def _mk_event(v_type, timestamp, ts_str, bbox, conf=0.0, meta=None) -> dict:
        e = {
            "type": v_type,
            "ts": ts_str,
            "timestamp": round(timestamp, 2),
            "frame": int(timestamp * 30),   # approx
            "bbox": [int(v) for v in bbox],
            "conf": round(conf, 3),
        }
        if meta:
            e.update(meta)
        return e

    def _in_cooldown(self, key, timestamp) -> bool:
        last = self._cooldown.get(key, -999)
        return (timestamp - last) < self._cooldown_sec

    def _set_cooldown(self, key, timestamp):
        self._cooldown[key] = timestamp
