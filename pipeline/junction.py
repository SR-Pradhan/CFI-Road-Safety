"""
JunctionDetector — classifies road intersections from dashcam frames.

Approach:
  1. Road segmentation via adaptive thresholding on the lower half of frame.
  2. Hough line detection to find road edge directions.
  3. Vanishing point analysis — multiple vanishing points suggest junction.
  4. Scene-change suppression — debounce repeated detections of same junction.

Types detected:
  T_JUNCTION  | X_JUNCTION | Y_JUNCTION | ROUNDABOUT | FLYOVER

Limitation:
  This heuristic approach works reasonably on clear daylight footage.
  A scene-classification CNN (EfficientNet fine-tuned on road junction images)
  would give substantially better precision. See README for dataset sources.
"""

import cv2
import numpy as np
from collections import deque


# Minimum seconds between successive junction detections
JUNCTION_COOLDOWN_SEC = 5.0


class JunctionDetector:
    def __init__(self):
        self._last_junction_ts = -999.0
        self._line_history = deque(maxlen=10)

    def detect(self, frame, curr_gray, prev_gray, timestamp, ts_str) -> dict | None:
        if (timestamp - self._last_junction_ts) < JUNCTION_COOLDOWN_SEC:
            return None

        h, w = curr_gray.shape
        # Focus on lower 60% of frame — road structure
        roi = curr_gray[int(h * 0.40):, :]

        # Edge detection
        edges = cv2.Canny(roi, threshold1=50, threshold2=150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1, theta=np.pi / 180,
            threshold=50,
            minLineLength=60,
            maxLineGap=15
        )

        if lines is None or len(lines) < 4:
            return None

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            angles.append(angle)

        self._line_history.append(len(lines))

        junction_type = self._classify_junction(angles, lines, frame, h, w)
        if junction_type is None:
            return None

        self._last_junction_ts = timestamp
        return {
            "junction_type": junction_type,
            "ts": ts_str,
            "timestamp": round(timestamp, 2),
            "frame": int(timestamp * 30),
            "line_count": len(lines),
        }

    def _classify_junction(self, angles, lines, frame, h, w) -> str | None:
        """
        Classify based on dominant angle clusters and scene features.
        """
        # Bin angles into clusters (horizontal ~0°/180°, vertical ~90°, diagonal)
        horizontal = [a for a in angles if a < 25 or a > 155]
        vertical   = [a for a in angles if 65 < a < 115]
        diagonal   = [a for a in angles if 25 <= a <= 65 or 115 <= a <= 155]

        n_h = len(horizontal)
        n_v = len(vertical)
        n_d = len(diagonal)
        total = len(angles)

        # Roundabout heuristic: detect circular-ish road contour
        if self._detect_roundabout(frame, h, w):
            return "ROUNDABOUT"

        # Flyover: sudden disappearance/appearance of road (road goes below/above)
        if self._detect_flyover(frame, h, w):
            return "FLYOVER"

        # Need significant branching to call anything a junction
        if total < 8:
            return None

        # X-junction (4-way): strong lines in both H and V directions
        if n_h >= 3 and n_v >= 3:
            return "X_JUNCTION"

        # T-junction: strong H lines + one strong V or D group
        if n_h >= 3 and (n_v >= 1 or n_d >= 2):
            return "T_JUNCTION"

        # Y-junction: strong diagonal lines diverging
        if n_d >= 4 and n_h < 3:
            return "Y_JUNCTION"

        # Fallback T-junction if enough lines and something changes
        if total >= 12:
            return "T_JUNCTION"

        return None

    def _detect_roundabout(self, frame, h, w) -> bool:
        """
        Look for large circular contour in the road region.
        """
        roi = frame[int(h * 0.3):, :]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_roi, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1, minDist=50,
            param1=50, param2=30,
            minRadius=30, maxRadius=min(h, w) // 3,
        )
        return circles is not None and len(circles[0]) >= 1

    def _detect_flyover(self, frame, h, w) -> bool:
        """
        Heuristic: sharp horizontal edge spanning >60% of frame width
        in the upper-middle portion → possible flyover/underpass structure.
        """
        roi_gray = cv2.cvtColor(frame[int(h*0.2):int(h*0.5), :], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(roi_gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                                minLineLength=int(w * 0.6), maxLineGap=20)
        return lines is not None and len(lines) >= 1
