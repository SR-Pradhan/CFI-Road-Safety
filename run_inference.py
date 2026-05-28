"""
CFI Road Safety Analysis — Calibrated Inference Pipeline
Bangalore dashcam video: 1280x720 @ 30fps, ~70 minutes

Detection strategy:
  Vehicles   : MOG2 background subtraction + contour sizing rules
  Violations : Multi-frame confirmation + per-track cooldowns (10s)
  Junctions  : Optical flow divergence + vert/diag line spike + 40s cooldown
  Tracking   : IoU centroid tracker, dedup across frames
"""

import cv2, numpy as np, json, time, os, sys
from collections import defaultdict, deque
from pathlib import Path

VIDEO  = 'Bangalore_City_Drive.mp4'
OUTDIR = './output'
SKIP   = 10

def fmt(s):
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"

# ── TRACKER ──────────────────────────────────────────────────────────────────
class Tracker:
    def __init__(self, maxgone=10):
        self.nid = 0; self.objs = {}; self.gone = {}
        self.all = {}; self.maxgone = maxgone

    def update(self, dets, ts):
        if not dets:
            for oid in list(self.gone):
                self.gone[oid] += 1
                if self.gone[oid] > self.maxgone:
                    del self.objs[oid]; del self.gone[oid]
            return {}
        if not self.objs:
            for bb, cat in dets:
                self.objs[self.nid] = (bb, cat)
                self.gone[self.nid] = 0
                self.all[self.nid]  = {"cat": cat, "first_ts": ts, "last_ts": ts}
                self.nid += 1
            return dict(self.objs)
        matched = set(); result = {}
        for oid, (obb, ocat) in list(self.objs.items()):
            best_iou, best_i = 0.18, -1
            for i, (bb, _) in enumerate(dets):
                if i in matched: continue
                iou = _iou(obb, bb)
                if iou > best_iou: best_iou, best_i = iou, i
            if best_i >= 0:
                bb, cat = dets[best_i]
                self.objs[oid] = (bb, cat); self.gone[oid] = 0
                self.all[oid]["last_ts"] = ts; matched.add(best_i)
                result[oid] = (bb, cat)
            else:
                self.gone[oid] += 1
                if self.gone[oid] > self.maxgone:
                    del self.objs[oid]; del self.gone[oid]
        for i, (bb, cat) in enumerate(dets):
            if i not in matched:
                self.objs[self.nid] = (bb, cat); self.gone[self.nid] = 0
                self.all[self.nid] = {"cat": cat, "first_ts": ts, "last_ts": ts}
                self.nid += 1
        return result

def _iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1),max(ay1,by1)
    ix2,iy2 = min(ax2,bx2),min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if not inter: return 0.0
    return inter/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)

# ── VEHICLE DETECTION ────────────────────────────────────────────────────────
def classify_bbox(x1,y1,x2,y2,H,W):
    w,h = x2-x1, y2-y1
    # Scale min area with resolution so 2W aren't lost at 360p
    scale = (H * W) / (720 * 1280)
    min_area = max(250, int(1500 * scale))
    if w*h < min_area or w*h > H*W*0.30: return None
    if y1 < H*0.22 or y2 > H*0.93: return None
    rel = w*h/(H*W)
    rw  = w/W
    ar  = w / max(h, 1)
    # Relative thresholds calibrated to work at 360p, 720p, 1080p:
    # At 360p: motorcycle ~30x50px=rel0.007, car ~80x60px=rel0.026, bus ~160x80px=rel0.056
    if rel > 0.050 or rw > 0.16: return "HMV"
    if rel > 0.012 or (ar > 1.15 and rel > 0.006): return "LMV"
    if rel >= 0.0022: return "2W"
    return None

def detect_vehicles(frame, fgmask, H, W):
    fg = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8))
    fg = cv2.morphologyEx(fg,     cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for cnt in cnts:
        scale = (H * W) / (720 * 1280)
        if cv2.contourArea(cnt) < max(250, int(1500 * scale)): continue
        rx,ry,rw,rh = cv2.boundingRect(cnt)
        cat = classify_bbox(rx,ry,rx+rw,ry+rh,H,W)
        if cat: dets.append(((rx,ry,rx+rw,ry+rh), cat))
    return dets

# ── VIOLATION ENGINE ─────────────────────────────────────────────────────────
class Violations:
    def __init__(self):
        self.cd = {}          # (type,oid) → last_ts
        self.cds   = 10.0     # 10-second cooldown per track
        self.pend  = defaultdict(int)   # pending[key] = consecutive frames seen
        self.CONFIRM = 2
        self.tl = "unk"
        self.last_persons = []
        self.person_frame = -999

    def update_persons(self, frame, fidx):
        """
        Resolution-independent rider detection using foreground blobs on 2W bboxes.
        Works at 360p where HOG fails (needs min ~64x128px window).
        Approach: within each 2W vehicle bbox upper region, count distinct
        vertical blobs — each represents a seated rider.
        """
        if fidx - self.person_frame < 4: return
        # persons are populated per-2W-vehicle in check(), not globally
        self.person_frame = fidx

    def update_tl(self, frame, H, W):
        """
        Strict traffic light detection:
        - Only look in top 35% of frame (TL is always above road level)
        - Further restrict to centre-right column (TL position in India)
        - Require a SMALL isolated blob (actual light ~8-25px radius at 360p)
          not a diffuse glow — eliminates brake lights, red cars, signboards
        """
        # TL region: top 35% of frame, centre 60% width
        ry2 = int(H * 0.35)
        rx1, rx2 = int(W * 0.20), int(W * 0.80)
        roi_bgr = frame[:ry2, rx1:rx2]
        roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        r1 = cv2.inRange(roi, np.array([0,  150, 100]), np.array([10, 255,255]))
        r2 = cv2.inRange(roi, np.array([165,150, 100]), np.array([180,255,255]))
        gn = cv2.inRange(roi, np.array([42,  80,  80]), np.array([88, 255,255]))
        rm = cv2.bitwise_or(r1, r2)

        def has_small_blob(mask, min_px=20, max_px=900):
            """True if mask has at least one tight circular-ish blob (actual light)."""
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                a = cv2.contourArea(cnt)
                if min_px < a < max_px:
                    _,_,cw,ch = cv2.boundingRect(cnt)
                    # Aspect ratio close to square (actual light is circular)
                    if 0.4 < cw/max(ch,1) < 2.5:
                        return True
            return False

        red_blob   = has_small_blob(rm)
        green_blob = has_small_blob(gn)

        if red_blob and not green_blob:     self.tl = "red"
        elif green_blob and not red_blob:   self.tl = "green"
        else:                               self.tl = "unk"

    def check(self, frame, vehicles, prev_gray, curr_gray, ts, ts_str, fidx):
        H,W = frame.shape[:2]
        self.update_tl(frame, H, W)
        self.update_persons(frame, fidx)
        events = []

        for oid, (bbox, cat) in vehicles.items():
            x1,y1,x2,y2 = bbox

            # ── helmet-less / triple riding (2W only) ───────────
            if cat == "2W":
                riders = self._count_riders(frame, bbox)

                if len(riders) >= 3:
                    key = ("triple_riding", oid)
                    self.pend[key] += 1
                    if self.pend[key] >= self.CONFIRM and not self._incd(key, ts):
                        events.append(self._mk("triple_riding",ts,ts_str,bbox,0.72,
                                               {"riders":len(riders)}))
                        self._setcd(key, ts); self.pend[key] = 0
                else:
                    self.pend[("triple_riding",oid)] = 0

                for rider_bbox in riders:
                    key = ("helmet_less", oid)
                    if not self._has_helmet(frame, rider_bbox):
                        self.pend[key] += 1
                        # Require 3 consecutive sampled frames of no-helmet evidence
                        if self.pend[key] >= 3 and not self._incd(key, ts):
                            events.append(self._mk("helmet_less",ts,ts_str,rider_bbox,0.61))
                            self._setcd(key, ts); self.pend[key] = 0
                    else:
                        self.pend[key] = 0

            # ── phone use ────────────────────────────────────────
            key = ("phone_use", oid)
            if self._phone_visible(frame, bbox, H, W):
                self.pend[key] += 1
                if self.pend[key] >= 3 and not self._incd(key, ts):
                    events.append(self._mk("phone_use",ts,ts_str,bbox,0.56))
                    self._setcd(key, ts); self.pend[key] = 0
            else:
                self.pend[key] = 0

            # ── wrong-side (optical flow) ─────────────────────────
            if prev_gray is not None:
                key = ("wrong_side", oid)
                dy = self._flow_dy(bbox, prev_gray, curr_gray)
                cx = (x1+x2)/2
                in_centre = W*0.30 < cx < W*0.70
                # Threshold -8.0 (was -5.5): only flag strong oncoming motion
                # CONFIRM raised to 3: must persist 3 sampled frames (~1 second)
                if dy is not None and dy < -8.0 and in_centre:
                    self.pend[key] += 1
                    if self.pend[key] >= 3 and not self._incd(key, ts):
                        events.append(self._mk("wrong_side",ts,ts_str,bbox,
                                               min(0.88,abs(dy)/12),{"flow_dy":round(dy,2)}))
                        self._setcd(key, ts); self.pend[key] = 0
                else:
                    self.pend[key] = 0

            # ── signal jump ───────────────────────────────────────
            if self.tl == "red":
                key = ("signal_jump", oid)
                stop_y = int(H * 0.58)
                # Must cross stop line by at least 5% of frame height to count
                if y2 > stop_y + int(H*0.05) and y1 < H*0.70:
                    self.pend[key] += 1
                    if self.pend[key] >= self.CONFIRM and not self._incd(key, ts):
                        events.append(self._mk("signal_jump",ts,ts_str,bbox,0.68))
                        self._setcd(key, ts); self.pend[key] = 0
                else:
                    self.pend[key] = 0

        return events

    def _count_riders(self, frame, bbox):
        """
        Count riders on a 2W by finding distinct vertical blobs in the upper
        portion of the vehicle bounding box. Works at any resolution.

        Returns list of per-rider bboxes [x1,y1,x2,y2].
        """
        x1,y1,x2,y2 = [int(v) for v in bbox]
        bw, bh = x2-x1, y2-y1
        if bw < 8 or bh < 10: return []

        # Upper 70% of 2W bbox contains riders (lower 30% is wheels/frame)
        roi_y2 = y1 + int(bh * 0.72)
        roi = frame[y1:roi_y2, x1:x2]
        if roi.size == 0: return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Edge-based segmentation: find vertical structures (heads + torsos)
        edges = cv2.Canny(gray, 30, 90)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return []

        # Each rider blob must be meaningfully large relative to the 2W box
        # Stricter than before to avoid noise blobs being counted as riders
        min_w = max(6, int(bw * 0.22))
        min_h = max(8, int(bh * 0.28))
        min_area = min_w * min_h

        riders = []
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(cnt) < min_area: continue
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rw < min_w or rh < min_h: continue
            # Aspect ratio check: a rider blob should be roughly upright
            if rw / max(rh, 1) > 3.0: continue  # too wide = road noise, not a person
            # Convert back to full-frame coords
            riders.append([x1+rx, y1+ry, x1+rx+rw, y1+ry+rh])

        # Merge overlapping rider blobs (same person detected by multiple contours)
        merged = []
        for r in riders:
            found = False
            for i, m in enumerate(merged):
                if _iou(r, m) > 0.30:
                    merged[i] = [min(r[0],m[0]),min(r[1],m[1]),
                                  max(r[2],m[2]),max(r[3],m[3])]
                    found = True; break
            if not found:
                merged.append(r)

        return merged[:4]  # max 4 riders per frame (sanity cap)

    def _has_helmet(self, frame, p):
        """
        Detects helmet presence using skin-tone analysis on head ROI.
        Logic: bare head = skin colour visible → no helmet.
               helmeted head = no skin → has helmet (or indeterminate → assume yes).
        Conservative bias: only flag helmet-less when skin tone is CLEARLY dominant.
        This minimises false positives at low resolutions where head ROI is tiny.
        """
        x1,y1,x2,y2 = [int(v) for v in p]
        # Use top 30% of rider bbox as head region
        hh = max(1, int((y2-y1) * 0.30))
        head = frame[y1:y1+hh, max(0,x1):min(frame.shape[1],x2)]

        # Need a minimum meaningful ROI — if too small, can't determine → assume helmet
        if head.shape[0] < 6 or head.shape[1] < 6 or head.size < 150:
            return True

        # Skin tone in YCrCb — robust across lighting conditions
        ycrcb = cv2.cvtColor(head, cv2.COLOR_BGR2YCrCb)
        # Skin range: Cr 133-173, Cb 77-127 (standard skin detection range)
        skin_mask = cv2.inRange(ycrcb,
                                np.array([0,  133, 77]),
                                np.array([255,173,127]))
        total_px = head.shape[0] * head.shape[1]
        skin_ratio = cv2.countNonZero(skin_mask) / max(total_px, 1)

        # Only flag as helmet-less if >28% of head region is skin tone
        # High threshold = conservative = fewer false positives
        return skin_ratio < 0.28

    def _phone_visible(self, frame, bbox, H, W):
        x1,y1,x2,y2 = [int(v) for v in bbox]
        # Only check LMV (cars) — phones rarely visible on 2W/HMV at road distance
        vw, vh = x2-x1, y2-y1
        if vw * vh < max(400, int(3000 * (H*W)/(720*1280))): return False
        mid_y = (y1+y2)//2
        rx1, rx2 = (x1+x2)//2, x2
        roi = frame[y1:mid_y, rx1:rx2]
        if roi.shape[0] < 10 or roi.shape[1] < 10: return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Require very bright (screen-bright) AND rectangular - threshold raised to 210
        _, br = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
        cnts,_ = cv2.findContours(br, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale = (H*W)/(720*1280)
        for cnt in cnts:
            a = cv2.contourArea(cnt)
            min_a = max(80, int(300 * scale))
            max_a = max(500, int(3000 * scale))
            if min_a < a < max_a:
                rx,ry,rw,rh = cv2.boundingRect(cnt)
                ar = rw/max(rh,1)
                # Phone aspect: portrait (0.4-0.8) or landscape (1.4-2.2)
                if (0.38 < ar < 0.82 or 1.4 < ar < 2.3) and min(rw,rh) > 8: return True
        return False

    def _flow_dy(self, bbox, prev_gray, curr_gray):
        x1,y1,x2,y2 = [int(v) for v in bbox]
        roi = prev_gray[y1:y2, x1:x2]
        if roi.shape[0] < 10 or roi.shape[1] < 10: return None
        pts = cv2.goodFeaturesToTrack(roi, 12, 0.3, 7)
        if pts is None or len(pts) < 3: return None
        pts = pts.copy(); pts[:,:,0]+=x1; pts[:,:,1]+=y1
        npts,st,_ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray,
                                              pts.astype(np.float32), None,
                                              winSize=(15,15), maxLevel=2)
        if npts is None: return None
        flat = st.flatten()
        good_o = pts[flat==1]; good_n = npts[flat==1]
        if len(good_o) < 2: return None
        if good_o.ndim == 3:
            return float(np.mean(good_n[:,0,1] - good_o[:,0,1]))
        return float(np.mean(good_n[:,1] - good_o[:,1]))

    @staticmethod
    def _mk(vtype,ts,ts_str,bbox,conf,extra=None):
        e = {"type":vtype,"ts":ts_str,"timestamp":round(ts,2),
             "frame":int(ts*30),"bbox":[int(v) for v in bbox],"conf":conf}
        if extra: e.update(extra)
        return e

    def _incd(self,k,ts): return (ts-self.cd.get(k,-999)) < self.cds
    def _setcd(self,k,ts): self.cd[k] = ts


# ── JUNCTION DETECTION ───────────────────────────────────────────────────────
class Junctions:
    def __init__(self):
        self.last_ts = -999.0
        self.COOLDOWN = 40.0          # min 40s between junctions
        self.vert_buf = deque(maxlen=6)
        self.diag_buf = deque(maxlen=6)
        self.flow_buf  = deque(maxlen=5)  # mean optical flow magnitude per frame

    def detect(self, frame, gray, prev_gray, ts, ts_str):
        if (ts - self.last_ts) < self.COOLDOWN:
            return None

        H,W = gray.shape
        # Road region only
        road = gray[int(H*0.38):int(H*0.88), :]
        edges = cv2.Canny(road, 65, 155, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 75,
                                minLineLength=75, maxLineGap=10)

        if lines is None:
            self.vert_buf.append(0); self.diag_buf.append(0)
            return None

        angles = [np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]))%180
                  for l in lines]
        vert = sum(1 for a in angles if 55 < a < 125)
        diag = sum(1 for a in angles if (22<=a<=55) or (125<=a<=158))
        horiz= sum(1 for a in angles if a<22 or a>158)

        # Optical flow spread (high spread = many motion directions = junction)
        flow_spread = 0.0
        if prev_gray is not None:
            samp = road
            pts = cv2.goodFeaturesToTrack(samp, 80, 0.3, 8)
            if pts is not None:
                pts_full = pts.copy()
                pts_full[:,:,1] += int(H*0.38)
                npts,st,_ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, pts_full.astype(np.float32), None,
                    winSize=(15,15), maxLevel=2)
                if npts is not None:
                    flat = st.flatten()
                    go = pts_full[flat==1]; gn = npts[flat==1]
                    if len(go) > 5:
                        if go.ndim == 3:
                            dx = gn[:,0,0]-go[:,0,0]; dy = gn[:,0,1]-go[:,0,1]
                        else:
                            dx = gn[:,0]-go[:,0]; dy = gn[:,1]-go[:,1]
                        angles_flow = np.arctan2(dy,dx)
                        flow_spread = float(np.std(angles_flow))

        self.vert_buf.append(vert)
        self.diag_buf.append(diag)
        self.flow_buf.append(flow_spread)

        mean_vert_prev = np.mean(list(self.vert_buf)[:-1]) if len(self.vert_buf)>1 else 0
        mean_flow_prev = np.mean(list(self.flow_buf)[:-1]) if len(self.flow_buf)>1 else 0

        # Junction conditions — ALL require multi-factor confirmation
        vert_spike = vert > max(mean_vert_prev*2.0, 5) and vert >= 5
        flow_spike  = flow_spread > max(mean_flow_prev*1.6, 0.4) and flow_spread > 0.5
        x_sig = horiz >= 6 and vert >= 5 and diag >= 3
        t_sig = horiz >= 10 and vert >= 4 and diag >= 1

        jtype = None
        if x_sig:                  jtype = "X_JUNCTION"
        elif t_sig:                jtype = "T_JUNCTION"
        elif vert_spike and flow_spike and diag >= 2: jtype = "T_JUNCTION"
        elif diag >= 6 and vert >= 3 and horiz < 8:  jtype = "Y_JUNCTION"

        # Roundabout: strict large-circle detection
        if jtype is None:
            blur = cv2.GaussianBlur(road, (11,11), 2)
            circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 80,
                                       param1=90, param2=60,
                                       minRadius=55, maxRadius=min(H,W)//4)
            if circles is not None and len(circles[0]) == 1:
                jtype = "ROUNDABOUT"

        # Flyover: 3+ very long horizontal lines spanning >70% width
        # AND they must be in upper-road region (y < 40% of road ROI = ~65% of frame)
        # Lane markings span full width too, so require them in UPPER portion of road view
        if jtype is None:
            long_h = [l for l in lines
                      if abs(l[0][3]-l[0][1]) < 8          # nearly horizontal
                      and abs(l[0][2]-l[0][0]) > W*0.68    # span >68% frame width
                      and l[0][1] < int(H*0.38)*0.35]      # upper road ROI only
            if len(long_h) >= 3: jtype = "FLYOVER"

        if jtype is None:
            return None

        self.last_ts = ts
        return {"junction_type": jtype, "ts": ts_str, "timestamp": round(ts,2),
                "frame": int(ts*30), "vert_lines": vert, "diag_lines": diag,
                "flow_spread": round(flow_spread,3)}


# ── MAIN LOOP ────────────────────────────────────────────────────────────────
def run():
    cap = cv2.VideoCapture(VIDEO)
    FPS   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    DUR   = TOTAL/FPS

    print(f"Video : {Path(VIDEO).name}")
    print(f"       {W}x{H} @ {FPS:.0f}fps | {DUR/60:.1f} min | {TOTAL} frames")
    print(f"       Sampling every {SKIP} frames ({FPS/SKIP:.1f} fps effective)")
    print("Running...\n")

    bg = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=45, detectShadows=False)
    tracker = Tracker(maxgone=12)
    veng    = Violations()
    jdet    = Junctions()

    v_events = defaultdict(list)
    j_events = []
    density  = []

    fidx = 0
    prev_gray = None
    samp_count = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        fidx += 1
        if fidx % SKIP != 0: continue
        samp_count += 1

        ts    = fidx / FPS
        ts_str= fmt(ts)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask= bg.apply(frame)

        # ── Vehicles ────────────────────────────────────────────
        raw = detect_vehicles(frame, fgmask, H, W)
        tracked = tracker.update(raw, ts)
        density.append({"timestamp": round(ts,2), "ts_str": ts_str,
                        "count": len(tracked)})

        # ── Violations ──────────────────────────────────────────
        evs = veng.check(frame, tracked, prev_gray, gray, ts, ts_str, samp_count)
        for e in evs:
            v_events[e["type"]].append(e)

        # ── Junctions ────────────────────────────────────────────
        j = jdet.detect(frame, gray, prev_gray, ts, ts_str)
        if j:
            j_events.append(j)
            print(f"  [{ts_str}] Junction: {j['junction_type']} (V={j['vert_lines']} D={j['diag_lines']})")

        prev_gray = gray

        # Progress every ~5 min
        if samp_count % 600 == 0:
            pct = fidx/TOTAL*100
            ela = time.time()-t0
            eta = ela/max(pct/100,0.01) - ela
            total_v = sum(len(x) for x in v_events.values())
            print(f"  [{ts_str}] {pct:.0f}% done | vehicles: {tracker.nid} | violations: {total_v} | ETA: {eta/60:.1f}min")

    cap.release()
    elapsed = time.time()-t0
    print(f"\nDone in {elapsed/60:.1f} min")

    # ── Compile ─────────────────────────────────────────────────
    LABELS = {
        "helmet_less":"Helmet-less Riding","wrong_side":"Wrong-side Driving",
        "signal_jump":"Signal Jumping","phone_use":"Mobile Phone Use",
        "triple_riding":"Triple Riding",
    }
    violations_out = {}
    for key, label in LABELS.items():
        evs = v_events.get(key, [])
        top3 = sorted(evs, key=lambda e: e.get("conf",0), reverse=True)[:3]
        violations_out[key] = {
            "label": label, "total_count": len(evs),
            "instances": [{"ts":e["ts"],"timestamp":e["timestamp"],
                           "frame":e["frame"],"conf":e["conf"]} for e in evs],
            "top3_annotations": [{"ts":e["ts"],"frame":e["frame"],
                                   "bbox":e.get("bbox"),"conf":e["conf"]} for e in top3],
        }

    cat_counts = defaultdict(int)
    for t in tracker.all.values(): cat_counts[t["cat"]] += 1
    total_v = max(sum(cat_counts.values()),1)
    vehicles_out = {
        "total_unique": sum(cat_counts.values()),
        "categories": {cat:{"count":cnt,"pct":round(cnt/total_v*100,1)}
                       for cat,cnt in cat_counts.items()}
    }

    type_counts = defaultdict(int)
    for j in j_events: type_counts[j["junction_type"]] += 1
    junctions_out = {
        "total_count": len(j_events),
        "type_breakdown": dict(type_counts),
        "instances": j_events,
    }

    results = {
        "meta": {
            "video": Path(VIDEO).name,
            "fps": round(FPS,1), "total_frames": TOTAL,
            "duration_seconds": round(DUR,1), "duration_hms": fmt(DUR),
            "resolution": f"{W}x{H}", "frame_skip": SKIP,
            "method": "Classical CV — MOG2+HOG+OptFlow+Hough",
            "processing_time_min": round(elapsed/60, 1),
        },
        "violations": violations_out,
        "junctions": junctions_out,
        "vehicles": vehicles_out,
        "density_timeline": density,
    }

    out_path = os.path.join(OUTDIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*55)
    print("RESULTS SUMMARY")
    print("="*55)
    tot_violations = sum(v["total_count"] for v in violations_out.values())
    print(f"Total violations : {tot_violations}")
    for k,v in violations_out.items():
        if v["total_count"]: print(f"  {v['label']:28s}: {v['total_count']}")
    print(f"Total junctions  : {junctions_out['total_count']}")
    for k,cnt in junctions_out["type_breakdown"].items():
        print(f"  {k:20s}: {cnt}")
    print(f"Total vehicles   : {vehicles_out['total_unique']}")
    for cat,d in vehicles_out["categories"].items():
        print(f"  {cat:8s}: {d['count']} ({d['pct']}%)")
    print(f"\nJSON → {out_path}")
    print("="*55)
    return results

if __name__ == "__main__":
    run()