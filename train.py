"""Train/test a pose-phase classifier from labelled images and videos.

Dataset layout (one folder per class):

    data/pose_training/
      squat_standing/
      squat_bottom/
      pushup_up/
      pushup_bottom/
      curl_left_extended/
      curl_left_flexed/
      curl_right_extended/
      curl_right_flexed/
      other/

Each class folder may contain JPG/PNG/WEBP images and MP4/AVI/MOV/MKV videos.
The split is performed at *source-file* level before video frames are expanded,
which avoids leaking adjacent frames from the same video into train and test.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from helper import coNguoi, layDiem
from pose_learning import FEATURE_VERSION, RECOMMENDED_LABELS, build_mlp, normalize_yolo_pose
from pose_model import get_pose_model, run_pose_inference
from config import MODELPATH

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def discover_sources(dataset_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    if not dataset_root.is_dir():
        return result
    for class_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        sources = [
            path for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        ]
        if sources:
            result[class_dir.name] = sources
    return result


def split_sources(sources: dict[str, list[Path]], seed: int = 20260809):
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    for label, files in sorted(sources.items()):
        files = list(files)
        rng.shuffle(files)
        n = len(files)
        if n == 1:
            split["train"].append((label, files[0]))
            continue
        if n == 2:
            split["train"].append((label, files[0]))
            split["test"].append((label, files[1]))
            continue
        test_n = max(1, int(round(n * 0.20)))
        val_n = max(1, int(round(n * 0.15))) if n >= 5 else 1
        train_n = max(1, n - test_n - val_n)
        while train_n + val_n + test_n > n and val_n > 0:
            val_n -= 1
        train_files = files[:train_n]
        val_files = files[train_n:train_n + val_n]
        test_files = files[train_n + val_n:]
        split["train"].extend((label, path) for path in train_files)
        split["val"].extend((label, path) for path in val_files)
        split["test"].extend((label, path) for path in test_files)
    return split


def _frame_feature(model, frame):
    results = run_pose_inference(model, frame)
    if not results or not coNguoi(results):
        return None
    points, conf = layDiem(results)
    try:
        return normalize_yolo_pose(points, conf)
    except ValueError:
        return None


def extract_source_features(model, path: Path, video_stride: int = 5, max_video_frames: int = 240):
    extension = path.suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path))
        if frame is None:
            return []
        feature = _frame_feature(model, frame)
        return [feature] if feature is not None else []

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    features = []
    index = 0
    try:
        while len(features) < max_video_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % max(1, video_stride) == 0:
                feature = _frame_feature(model, frame)
                if feature is not None:
                    features.append(feature)
            index += 1
    finally:
        capture.release()
    return features


def expand_split(split_items, model, label_to_index, video_stride):
    xs, ys = [], []
    source_stats = []
    for label, path in split_items:
        features = extract_source_features(model, path, video_stride=video_stride)
        source_stats.append({"label": label, "source": str(path), "frames": len(features)})
        xs.extend(features)
        ys.extend([label_to_index[label]] * len(features))
    if not xs:
        return np.empty((0, 51), dtype=np.float32), np.empty((0,), dtype=np.int64), source_stats
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), source_stats


def metrics(model, x, y, labels):
    if len(x) == 0:
        return {"samples": 0, "accuracy": None, "per_class": {}, "confusion_matrix": []}
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).float())
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    correct = int(np.sum(pred == y))
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, guess in zip(y, pred):
        matrix[int(truth), int(guess)] += 1
    per_class = {}
    for idx, label in enumerate(labels):
        total = int(matrix[idx].sum())
        per_class[label] = {
            "samples": total,
            "accuracy": round(float(matrix[idx, idx] / total), 4) if total else None,
        }
    return {
        "samples": int(len(y)),
        "accuracy": round(correct / max(1, len(y)), 4),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/pose_training")
    parser.add_argument("--output", default="models/pose_phase_classifier.pth")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--video-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated labels to train. Empty means every class folder in dataset.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.dataset)
    sources = discover_sources(root)
    requested_labels = [item.strip() for item in str(args.labels or "").split(",") if item.strip()]
    if requested_labels:
        wanted = set(requested_labels)
        sources = {label: files for label, files in sources.items() if label in wanted}
    if len(sources) < 2:
        print("Chưa đủ dữ liệu. Cần ít nhất 2 class có ảnh/video trong", root)
        print("Các label khuyến nghị:", ", ".join(RECOMMENDED_LABELS))
        raise SystemExit(2)

    labels = sorted(sources)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    split = split_sources(sources, seed=args.seed)
    print("Source split:", {name: len(items) for name, items in split.items()})

    pose_model = get_pose_model(MODELPATH)
    extracted = {}
    source_stats = {}
    for name in ("train", "val", "test"):
        x, y, stats = expand_split(split[name], pose_model, label_to_index, args.video_stride)
        extracted[name] = (x, y)
        source_stats[name] = stats
        print(f"{name}: {len(y)} pose frame")

    train_x, train_y = extracted["train"]
    if len(train_y) < max(8, len(labels) * 2):
        print("Dữ liệu train sau khi trích pose còn quá ít.")
        raise SystemExit(3)

    model = build_mlp(train_x.shape[1], len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()),
        batch_size=max(4, args.batch_size),
        shuffle=True,
    )

    best_state = None
    best_val = -1.0
    for epoch in range(1, max(1, args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        val = metrics(model, *extracted["val"], labels)
        val_acc = -1.0 if val["accuracy"] is None else float(val["accuracy"])
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"Epoch {epoch:02d}/{args.epochs}: loss={loss_sum:.4f}, val_acc={val['accuracy']}")

    if best_state is not None:
        model.load_state_dict(best_state)

    report = {
        "feature_version": FEATURE_VERSION,
        "labels": labels,
        "requested_labels": requested_labels,
        "source_counts": {label: len(files) for label, files in sources.items()},
        "split_sources": source_stats,
        "train": metrics(model, *extracted["train"], labels),
        "validation": metrics(model, *extracted["val"], labels),
        "test": metrics(model, *extracted["test"], labels),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "labels": labels,
        "input_dim": int(train_x.shape[1]),
        "feature_version": FEATURE_VERSION,
        "metrics": report,
    }, str(output))
    report_path = output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu model:", output)
    print("Đã lưu test metrics:", report_path)
    print("Test accuracy:", report["test"]["accuracy"])


if __name__ == "__main__":
    main()
