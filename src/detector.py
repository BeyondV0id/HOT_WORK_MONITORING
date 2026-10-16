import math
import numpy as np

CLASS_NAMES = {
    0: 'Flame/Fire',
    1: 'no gloves',
    2: 'gloves',
    3: 'sparks',
    4: 'Tools',
    5: 'person',
    6: 'Shoes',
    7: 'Safety-vest',
    8: 'Helmet',
    9: 'Artificial light'
}

HOT_WORK_CLASSES = {'Flame/Fire', 'sparks'}
PPE_CLASSES = {'Helmet', 'Safety-vest', 'gloves', 'Shoes'}
VIOLATION_CLASSES = {'no gloves'}

def get_box_center(box):
    """Calculates center point (cx, cy) of bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def euclidean_distance(pt1, pt2):
    """Calculates Euclidean distance between two 2D points."""
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

def point_in_box(point, box, margin=15):
    """Checks if point (px, py) is inside bounding box [x1, y1, x2, y2] with optional margin."""
    px, py = point
    x1, y1, x2, y2 = box
    return (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (y2 + margin)

def box_intersection_over_min_area(boxA, boxB):
    """Calculates overlap area divided by area of smaller box."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[0])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    minArea = min(boxAArea, boxBArea)
    
    return interArea / float(minArea) if minArea > 0 else 0.0

class HotWorkComplianceAnalyzer:
    """
    Analyzes object detections from YOLO model for Hot Work Safety Rules:
    - Rule 1: Detect active Hot Work (Flame/Fire or sparks).
    - Rule 2: Identify workers (person) in proximity to hot work zone.
    - Rule 3: Check mandatory PPE (Helmet, Safety-vest, gloves) for nearby workers.
    - Rule 4: Flag safety violation if worker lacks required PPE or displays explicit 'no gloves'.
    """
    def __init__(self, proximity_threshold=350, conf_threshold=0.25, required_ppe=None):
        self.proximity_threshold = proximity_threshold
        self.conf_threshold = conf_threshold
        self.required_ppe = required_ppe or ['Helmet', 'Safety-vest', 'gloves']

    def parse_ultralytics_results(self, results):
        """Converts Ultralytics YOLO Results object into standard list of dicts."""
        detections = []
        if len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].cpu().item())
            if conf < self.conf_threshold:
                continue
            cls_id = int(boxes.cls[i].cpu().item())
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            detections.append({
                'box': xyxy,
                'class_id': cls_id,
                'class_name': cls_name,
                'conf': conf
            })
        return detections

    def analyze(self, raw_detections):
        """
        Executes spatial compliance checks on frame detections.
        Can take either a list of detection dicts or Ultralytics Results object.
        """
        if hasattr(raw_detections, '__len__') and len(raw_detections) > 0 and hasattr(raw_detections[0], 'boxes'):
            detections = self.parse_ultralytics_results(raw_detections)
        else:
            detections = [d for d in raw_detections if d['conf'] >= self.conf_threshold]

        hot_work_items = []
        persons = []
        ppes = []
        explicit_violations = []

        for d in detections:
            cname = d['class_name']
            if cname in HOT_WORK_CLASSES:
                hot_work_items.append(d)
            elif cname == 'person':
                persons.append(d)
            elif cname in PPE_CLASSES:
                ppes.append(d)
            elif cname in VIOLATION_CLASSES:
                explicit_violations.append(d)

        is_hot_work_active = len(hot_work_items) > 0
        violations_found = []
        persons_in_danger_zone = 0

        if is_hot_work_active:
            for p_idx, person in enumerate(persons):
                p_box = person['box']
                p_center = get_box_center(p_box)

                # Find closest hot work source
                min_dist = float('inf')
                closest_hw = None
                for hw in hot_work_items:
                    hw_center = get_box_center(hw['box'])
                    dist = euclidean_distance(p_center, hw_center)
                    if dist < min_dist:
                        min_dist = dist
                        closest_hw = hw

                # Check if person is in proximity danger zone
                if min_dist <= self.proximity_threshold:
                    persons_in_danger_zone += 1
                    
                    # Identify PPE associated with this person
                    detected_person_ppe = set()
                    for ppe in ppes:
                        ppe_center = get_box_center(ppe['box'])
                        if point_in_box(ppe_center, p_box, margin=30) or box_intersection_over_min_area(ppe['box'], p_box) > 0.4:
                            detected_person_ppe.add(ppe['class_name'])

                    # Check for explicit violation items (e.g. 'no gloves')
                    has_no_gloves = False
                    for ex_v in explicit_violations:
                        v_center = get_box_center(ex_v['box'])
                        if point_in_box(v_center, p_box, margin=20):
                            has_no_gloves = True

                    # Evaluate missing required PPE
                    missing_ppe = []
                    for req in self.required_ppe:
                        if req == 'gloves':
                            if has_no_gloves or ('gloves' not in detected_person_ppe):
                                missing_ppe.append('gloves')
                        elif req not in detected_person_ppe:
                            missing_ppe.append(req)

                    if len(missing_ppe) > 0:
                        violations_found.append({
                            'person_id': p_idx,
                            'person_box': p_box,
                            'hot_work_class': closest_hw['class_name'],
                            'hot_work_box': closest_hw['box'],
                            'distance_px': round(min_dist, 1),
                            'missing_ppe': missing_ppe,
                            'detected_ppe': list(detected_person_ppe),
                            'has_no_gloves_flag': has_no_gloves,
                            'severity': 'HIGH' if 'Helmet' in missing_ppe or has_no_gloves else 'MEDIUM'
                        })

        return {
            'is_hot_work_active': is_hot_work_active,
            'hot_work_count': len(hot_work_items),
            'hot_work_items': hot_work_items,
            'total_persons': len(persons),
            'persons_in_danger_zone': persons_in_danger_zone,
            'all_detections': detections,
            'has_violation': len(violations_found) > 0,
            'violations': violations_found
        }
