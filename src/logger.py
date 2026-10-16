import os
import sys
import cv2
import json
import csv
import time
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class ViolationLogger:
    """
    Handles logging of Hot Work safety violations:
    - Draws visual overlays, bounding boxes, and alert banners on frame.
    - Adaptively scales annotations according to frame resolution to preserve aspect ratio & visual clarity.
    - Saves annotated snapshot images to `logs/snapshots/`.
    - Appends incident data to `logs/incidents.json` and `logs/incidents.csv`.
    - Manages cooldown timers to avoid duplicate alert flooding.
    """
    def __init__(self, log_dir="logs", cooldown_seconds=3.0):
        self.log_dir = log_dir
        self.snapshot_dir = os.path.join(log_dir, "snapshots")
        self.json_path = os.path.join(log_dir, "incidents.json")
        self.csv_path = os.path.join(log_dir, "incidents.csv")
        self.cooldown_seconds = cooldown_seconds
        self.last_log_time = 0.0

        os.makedirs(self.snapshot_dir, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        """Creates CSV header if file doesn't exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Frame_ID", "Hot_Work_Type", 
                    "Missing_PPE", "Distance_px", "Severity", "Snapshot_Path"
                ])

    def can_log(self):
        """Checks if cooldown period has elapsed since last logged violation."""
        return (time.time() - self.last_log_time) >= self.cooldown_seconds

    def draw_annotations(self, frame, analysis_result):
        """Draws bounding boxes, lines, and warning badges adaptively scaled for aspect ratio & resolution."""
        annotated = frame.copy()
        fh, fw = annotated.shape[:2]

        # Resolution scaling factor (base resolution = 1280x720)
        scale = max(0.6, min(fw / 1280.0, fh / 720.0))
        box_thick = max(2, int(3 * scale))
        line_thick = max(2, int(2 * scale))
        font_scale = max(0.5, 0.65 * scale)
        font_thick = max(1, int(2 * scale))

        # Color definitions (BGR)
        COLOR_HOTWORK = (0, 140, 255)   # Orange
        COLOR_VIOLATION = (0, 0, 255)   # Red
        COLOR_TEXT = (255, 255, 255)

        # 1. Draw Hot Work sources
        for hw in analysis_result['hot_work_items']:
            x1, y1, x2, y2 = map(int, hw['box'])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_HOTWORK, box_thick)
            label = f"HOT WORK: {hw['class_name']} ({hw['conf']:.2f})"
            lbl_pos_y = max(int(30 * scale), y1 - int(10 * scale))
            cv2.putText(annotated, label, (x1, lbl_pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_HOTWORK, font_thick)

        # 2. Draw other detections
        for d in analysis_result['all_detections']:
            if d['class_name'] in ['person', 'Flame/Fire', 'sparks']:
                continue
            x1, y1, x2, y2 = map(int, d['box'])
            color = (0, 0, 255) if d['class_name'] == 'no gloves' else (255, 200, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, max(1, int(1.5 * scale)))
            cv2.putText(annotated, f"{d['class_name']}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, color, max(1, font_thick - 1))

        # 3. Draw Violations / Workers
        for v in analysis_result['violations']:
            px1, py1, px2, py2 = map(int, v['person_box'])
            hx1, hy1, hx2, hy2 = map(int, v['hot_work_box'])
            
            pc = (int((px1 + px2) / 2), int((py1 + py2) / 2))
            hc = (int((hx1 + hx2) / 2), int((hy1 + hy2) / 2))

            # Proximity line
            cv2.line(annotated, pc, hc, COLOR_VIOLATION, line_thick, cv2.LINE_AA)
            dist_label = f"Dist: {v['distance_px']}px"
            mid_pt = (int((pc[0] + hc[0])/2), int((pc[1] + hc[1])/2))
            cv2.putText(annotated, dist_label, mid_pt,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 255, 255), font_thick)

            # Person box
            cv2.rectangle(annotated, (px1, py1), (px2, py2), COLOR_VIOLATION, box_thick)
            
            # Badge text
            missing_str = ", ".join(v['missing_ppe'])
            v_label = f"VIOLATION! Missing: {missing_str}"
            
            (tw, th), _ = cv2.getTextSize(v_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            bg_y1 = max(0, py1 - th - int(12 * scale))
            cv2.rectangle(annotated, (px1, bg_y1), (px1 + tw + int(12 * scale), py1), COLOR_VIOLATION, -1)
            cv2.putText(annotated, v_label, (px1 + int(6 * scale), max(th, py1 - int(6 * scale))),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, font_thick)

        # 4. Top Status Header Banner (Aspect Ratio Preserved)
        header_height = int(55 * scale)
        header_color = COLOR_VIOLATION if analysis_result['has_violation'] else (
            (0, 180, 0) if analysis_result['is_hot_work_active'] else (50, 50, 50)
        )
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (fw, header_height), header_color, -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        if analysis_result['has_violation']:
            status_text = f"ALERT: {len(analysis_result['violations'])} SAFETY VIOLATION(S) DETECTED"
        elif analysis_result['is_hot_work_active']:
            status_text = "HOT WORK ACTIVE - WORKERS COMPLIANT"
        else:
            status_text = "MONITORING - NO HOT WORK DETECTED"

        txt_y = int(36 * scale)
        cv2.putText(annotated, status_text, (int(20 * scale), txt_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, COLOR_TEXT, font_thick + 1)

        return annotated

    def log_violation(self, frame, analysis_result, frame_id=0):
        """Saves violation snapshot image and writes structured log records."""
        if not analysis_result['has_violation'] or not self.can_log():
            return None

        self.last_log_time = time.time()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        formatted_dt = datetime.now().isoformat()

        annotated_frame = self.draw_annotations(frame, analysis_result)
        snapshot_filename = f"violation_{timestamp_str}.jpg"
        snapshot_filepath = os.path.join(self.snapshot_dir, snapshot_filename)
        cv2.imwrite(snapshot_filepath, annotated_frame)

        log_records = []
        for v in analysis_result['violations']:
            record = {
                "timestamp": formatted_dt,
                "frame_id": frame_id,
                "hot_work_type": v['hot_work_class'],
                "missing_ppe": v['missing_ppe'],
                "distance_px": v['distance_px'],
                "severity": v['severity'],
                "snapshot_path": snapshot_filepath.replace("\\", "/")
            }
            log_records.append(record)

            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    record['timestamp'], record['frame_id'], record['hot_work_type'],
                    ";".join(record['missing_ppe']), record['distance_px'],
                    record['severity'], record['snapshot_path']
                ])

        existing_json = []
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    existing_json = json.load(f)
            except Exception:
                existing_json = []
        
        existing_json.extend(log_records)
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_json, f, indent=2)

        print(f"🚨 [VIOLATION LOGGED] Snapshot saved: {snapshot_filepath}")
        return snapshot_filepath
