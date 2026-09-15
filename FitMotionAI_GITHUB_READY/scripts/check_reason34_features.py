"""Offline preflight for Reason 3 + Reason 4 + emergency admin alerts."""
from pathlib import Path
import json
import tempfile
import sys

import numpy as np

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from pose_learning import BilateralCurlGuard, normalize_yolo_pose
from reference_motion import ReferenceMotionMatcher, load_reference_file

pts = np.array([[20+i*3, 30+(i%5)*7] for i in range(17)], dtype=np.float32)
pts[5] = [40,30]; pts[6] = [70,30]; pts[11] = [45,80]; pts[12] = [65,80]
conf = np.full(17, 0.95, dtype=np.float32)
assert normalize_yolo_pose(pts, conf).shape == (51,)

guard = BilateralCurlGuard('left')
guard.observe(150,160); guard.observe(145,90); guard.observe(148,155)
assert guard.wrong_side()

frames = [{"t":i/9,"knee_angle":170-i*4,"torso_angle":170-i} for i in range(10)]
payload = {"schema":"fitmotion.reference.v1","exercise":"squat","animation_key":"Squat","version":"mixamo-v1","source":{"type":"mixamo-blender"},"metrics":["knee_angle","torso_angle"],"metric_scales":{"knee_angle":40,"torso_angle":15},"min_score":55,"frames":frames}
with tempfile.TemporaryDirectory() as td:
    path=Path(td)/'ref.json'; path.write_text(json.dumps(payload),encoding='utf-8')
    load_reference_file(path,'squat')
    matcher=ReferenceMotionMatcher(str(path),'squat'); matcher.start_rep()
    for frame in frames: matcher.observe(frame)
    assert matcher.finish_rep().score >= 99

required = {
    'app.py':['admin_training_media_upload','admin_reference_upload','persist_emergency_alert_if_needed'],
    'templates/_admin_notification_bell.html':['setInterval(poll, 2500)','adminEmergencyToast'],
    'scripts/blender_export_mixamo_reference.py':['mixamorig:LeftArm','fitmotion.reference.v1'],
}
for rel, needles in required.items():
    data=(root/rel).read_text(encoding='utf-8')
    for needle in needles: assert needle in data, (rel,needle)
print('REASON34_PREFLIGHT_PASS')
