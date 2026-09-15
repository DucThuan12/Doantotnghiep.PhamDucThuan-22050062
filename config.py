MODELPATH = "yolov8n-pose.pt"
CAMERAINDEX = 0

KEYPOINTTHRESH = 0.30

# cooldown cho canh bao he thong
WARNINGCOOLDOWN = 10.0
WARNINGSHOWSEC = 2.0

# cooldown cho chup anh loi (rep sai)
ERRORCOOLDOWN = 2.0
ERRORSHOWSEC = 4.0

SHOWFPS = True
# Realtime pose performance tuning. Lower inference size reduces latency while
# preserving the original camera preview resolution.
POSE_IMGSZ = 416
POSE_CONF = 0.25
POSE_USE_HALF = True

