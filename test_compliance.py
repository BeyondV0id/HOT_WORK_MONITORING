import os
import sys
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.detector import HotWorkComplianceAnalyzer
from src.logger import ViolationLogger

def test_synthetic_compliance_logic():
    print("--- Test 1: Synthetic Detections Rule Verification ---")
    analyzer = HotWorkComplianceAnalyzer(proximity_threshold=300, conf_threshold=0.25)

    # Simulated scene 1: Hot Work (Flame/Fire) + Worker nearby with ONLY Safety-vest (Missing Helmet & gloves)
    synthetic_detections_violation = [
        {'box': [100, 100, 200, 200], 'class_id': 0, 'class_name': 'Flame/Fire', 'conf': 0.92},
        {'box': [220, 120, 350, 450], 'class_id': 5, 'class_name': 'person', 'conf': 0.88},
        {'box': [230, 200, 340, 350], 'class_id': 7, 'class_name': 'Safety-vest', 'conf': 0.85},
        {'box': [230, 280, 260, 310], 'class_id': 1, 'class_name': 'no gloves', 'conf': 0.79},
    ]

    result = analyzer.analyze(synthetic_detections_violation)
    
    assert result['is_hot_work_active'] is True, "Hot work should be detected"
    assert result['has_violation'] is True, "Violation should be flagged"
    assert len(result['violations']) == 1, "One worker in violation"
    
    v = result['violations'][0]
    assert 'Helmet' in v['missing_ppe'], "Helmet should be missing"
    assert 'gloves' in v['missing_ppe'], "gloves should be missing due to 'no gloves'"
    print("[PASS] Test 1 Passed: Violation correctly flagged for worker missing Helmet and wearing no gloves.")

    # Simulated scene 2: Worker fully compliant (Helmet + Safety-vest + gloves)
    synthetic_detections_compliant = [
        {'box': [100, 100, 200, 200], 'class_id': 0, 'class_name': 'Flame/Fire', 'conf': 0.95},
        {'box': [220, 120, 350, 450], 'class_id': 5, 'class_name': 'person', 'conf': 0.88},
        {'box': [245, 125, 305, 175], 'class_id': 8, 'class_name': 'Helmet', 'conf': 0.91},
        {'box': [230, 200, 340, 350], 'class_id': 7, 'class_name': 'Safety-vest', 'conf': 0.89},
        {'box': [230, 280, 260, 310], 'class_id': 2, 'class_name': 'gloves', 'conf': 0.82},
    ]

    result_comp = analyzer.analyze(synthetic_detections_compliant)
    assert result_comp['is_hot_work_active'] is True, "Hot work should be active"
    assert result_comp['has_violation'] is False, "No violation should be flagged for compliant worker"
    print("[PASS] Test 2 Passed: Compliant worker correctly passed with zero violations.")

def test_logger_and_snapshot():
    print("\n--- Test 2: Snapshot Rendering & Incident Logging Verification ---")
    analyzer = HotWorkComplianceAnalyzer(proximity_threshold=350)
    logger = ViolationLogger(log_dir="test_logs", cooldown_seconds=0.0)

    # Create dummy frame (640x480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Simulated violation detections
    detections = [
        {'box': [50, 50, 150, 150], 'class_id': 3, 'class_name': 'sparks', 'conf': 0.90},
        {'box': [120, 60, 280, 400], 'class_id': 5, 'class_name': 'person', 'conf': 0.87},
        {'box': [130, 150, 250, 300], 'class_id': 7, 'class_name': 'Safety-vest', 'conf': 0.80},
    ]

    analysis = analyzer.analyze(detections)
    snapshot_file = logger.log_violation(frame, analysis, frame_id=1)

    assert snapshot_file is not None, "Snapshot file path should be returned"
    assert os.path.exists(snapshot_file), f"Snapshot image must exist at {snapshot_file}"
    assert os.path.exists("test_logs/incidents.json"), "incidents.json should be created"
    assert os.path.exists("test_logs/incidents.csv"), "incidents.csv should be created"

    print(f"[PASS] Test 2 Passed: Snapshot rendered ({snapshot_file}) and logs created.")

if __name__ == "__main__":
    test_synthetic_compliance_logic()
    test_logger_and_snapshot()
    print("\n[SUCCESS] ALL COMPLIANCE LOGIC & LOGGING VERIFICATION TESTS PASSED!")
