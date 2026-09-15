import os
import csv
import time
import unicodedata
import cv2
import numpy as np


def tinhgoc(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    normba = np.linalg.norm(ba)
    normbc = np.linalg.norm(bc)

    if normba == 0 or normbc == 0:
        return 0.0

    cosine = np.dot(ba, bc) / (normba * normbc)
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine))
    return float(angle)


class FPSCounter:
    def __init__(self):
        self.prevtime = time.time()

    def get(self):
        now = time.time()
        fps = 1.0 / max(now - self.prevtime, 1e-6)
        self.prevtime = now
        return int(fps)


class WarningManager:
    def __init__(self, cooldown=10.0, showsec=2.0):
        self.cooldown = cooldown
        self.showsec = showsec
        self.lasttime = {}
        self.text = ""
        self.until = 0

    def trigger(self, key, text):
        now = time.time()
        if now - self.lasttime.get(key, 0) >= self.cooldown:
            self.lasttime[key] = now
            self.text = text
            self.until = now + self.showsec

    def draw(self, frame, x=30, y=390):
        if time.time() < self.until and self.text:
            vietchu(frame, self.text, x, y, (0, 0, 255), 0.75, 2)


class ResultManager:
    def __init__(self, showsec=4.0):
        self.showsec = showsec
        self.result = ""
        self.advice = ""
        self.until = 0

    def set(self, result, advice):
        self.result = result
        self.advice = advice
        self.until = time.time() + self.showsec

    def draw(self, frame, x=30, y=300):
        if time.time() < self.until and self.result:
            vietchu(frame, f"KET QUA: {self.result}", x, y, (0, 0, 255), 0.75, 2)
            vietchu(frame, f"KHUYEN NGHI: {self.advice}", x, y + 40, (0, 255, 255), 0.75, 2)


def taothumuc(path):
    os.makedirs(path, exist_ok=True)


def ghiloi(exercisename, errorcode, errortext, advicetext, imagepath):
    logdir = os.path.join("data", "logs")
    taothumuc(logdir)

    csvpath = os.path.join(logdir, "errorlog.csv")
    fileexists = os.path.exists(csvpath)

    with open(csvpath, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not fileexists:
            writer.writerow(["time", "exercise", "errorcode", "errortext", "advicetext", "imagepath"])

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            exercisename,
            errorcode,
            errortext,
            advicetext,
            imagepath
        ])


def to_ascii_overlay(text):
    """Convert Vietnamese/UI text to OpenCV-safe ASCII for camera overlays.

    ``cv2.putText`` uses Hershey fonts and does not render Vietnamese Unicode
    reliably. Keep the web UI/voice text in Vietnamese, but strip accents only
    at the final camera-overlay boundary so strings never appear as ``??``.
    """
    value = str(text or "")
    value = value.replace("Đ", "D").replace("đ", "d")
    normalized = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn").encode("ascii", "ignore").decode("ascii")


def vietchu(frame, text, x, y, color=(0, 255, 0), scale=0.8, thickness=2):
    cv2.putText(
        frame,
        to_ascii_overlay(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def coNguoi(results):
    return (
        results[0].keypoints is not None
        and results[0].keypoints.xy is not None
        and len(results[0].keypoints.xy) > 0
    )


def layDiem(results):
    """
    Chon nguoi co do tin cay keypoint trung binh cao nhat thay vi luon lay [0].
    Cac truong hop co nhieu nguoi / model tra ve nguoi khong on dinh se giam bi bat nham.
    """
    kp_xy_all = results[0].keypoints.xy.cpu().numpy()
    kp_conf_all = results[0].keypoints.conf.cpu().numpy()

    if len(kp_xy_all) == 1:
        return kp_xy_all[0], kp_conf_all[0]

    best_idx = 0
    best_score = -1.0
    for i in range(len(kp_conf_all)):
        conf = kp_conf_all[i]
        score = float(np.mean(conf))
        if score > best_score:
            best_score = score
            best_idx = i

    return kp_xy_all[best_idx], kp_conf_all[best_idx]


def thayKhau(confarray, jointidx, threshold=0.30):
    return confarray[jointidx] >= threshold


def veKhungThongTin(frame, exercisename, resulttext, advicetext, metriclines=None):
    img = frame.copy()

    h, w = img.shape[:2]
    box_h = 180 + (len(metriclines) * 30 if metriclines else 0)

    overlay = img.copy()
    cv2.rectangle(overlay, (20, 20), (w - 20, min(20 + box_h, h - 20)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    vietchu(img, f"BAI TAP: {exercisename.upper()}", 35, 55, (255, 255, 255), 0.9, 2)
    vietchu(img, f"KET QUA: {resulttext}", 35, 95, (0, 0, 255), 0.8, 2)
    vietchu(img, f"KHUYEN NGHI: {advicetext}", 35, 135, (0, 255, 255), 0.75, 2)
    vietchu(img, f"THOI GIAN: {time.strftime('%Y-%m-%d %H:%M:%S')}", 35, 175, (255, 255, 255), 0.65, 2)

    if metriclines:
        y = 215
        for line in metriclines:
            vietchu(img, line, 35, y, (0, 255, 0), 0.72, 2)
            y += 30

    return img


def luuanhbadrep(frame, exercisename, errorcode, errortext, advicetext, metriclines=None):
    folder = os.path.join("data", "errorimages", exercisename)
    taothumuc(folder)

    img = veKhungThongTin(frame, exercisename, errortext, advicetext, metriclines)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{errorcode}_{timestamp}.jpg"
    filepath = os.path.join(folder, filename)

    cv2.imwrite(filepath, img)
    return filepath


def save_bad_rep(frame, exercisename, errorcode, errortext, advicetext, metriclines, lastsave, cooldown):
    now = time.time()
    if now - lastsave >= cooldown:
        imagepath = luuanhbadrep(frame, exercisename, errorcode, errortext, advicetext, metriclines)
        ghiloi(exercisename, errorcode, errortext, advicetext, imagepath)
        return now
    return lastsave
