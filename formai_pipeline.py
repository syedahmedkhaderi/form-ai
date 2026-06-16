#!/usr/bin/env python3

# Environment check
import json
import math
import os
import random
import shutil
from datetime import date
from pathlib import Path

import coremltools as ct
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())


# Constants and contracts
SEED = 42
W = 32
DATA_DIR = Path("formai_data")

MANIFEST_COLUMNS = [
    "clip_id",
    "file",
    "exercise",
    "view",
    "label",
    "error_type",
    "subject_id",
    "fps",
    "num_frames",
    "notes",
]

SQUAT_FEATURE_ORDER = [
    "11x",
    "11y",
    "12x",
    "12y",
    "23x",
    "23y",
    "24x",
    "24y",
    "25x",
    "25y",
    "26x",
    "26y",
    "27x",
    "27y",
    "28x",
    "28y",
    "left_knee_angle",
    "right_knee_angle",
    "torso_lean",
    "left_valgus",
    "right_valgus",
]

CURL_FEATURE_ORDER = [
    "12x",
    "12y",
    "14x",
    "14y",
    "16x",
    "16y",
    "11x",
    "11y",
    "right_elbow_angle",
    "shoulder_tilt",
    "elbow_flare",
]

CLASS_LABELS = {
    "squat": ["good", "knee_valgus", "insufficient_depth", "back_rounding"],
    "curl": ["good", "swing", "partial_rom", "elbow_flare"],
}

ERROR_TO_CLASS = {
    "squat": {
        "none": 0,
        "knee_valgus": 1,
        "insufficient_depth": 2,
        "back_rounding": 3,
    },
    "curl": {
        "none": 0,
        "swing": 1,
        "partial_rom": 2,
        "elbow_flare": 3,
    },
}


# Reproducibility
def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Synthetic data generator
def _base_pose() -> np.ndarray:
    frame = np.zeros((33, 4), dtype=np.float32)
    frame[:, 0] = 0.5
    frame[:, 1] = 0.5
    frame[:, 2] = 0.0
    frame[:, 3] = 0.95

    coords = {
        11: (0.42, 0.28),
        12: (0.58, 0.28),
        13: (0.38, 0.42),
        14: (0.62, 0.42),
        15: (0.36, 0.56),
        16: (0.64, 0.56),
        23: (0.45, 0.55),
        24: (0.55, 0.55),
        25: (0.43, 0.72),
        26: (0.57, 0.72),
        27: (0.41, 0.92),
        28: (0.59, 0.92),
    }
    for idx, (x, y) in coords.items():
        frame[idx, 0] = x
        frame[idx, 1] = y
    return frame


def _make_squat_clip(t: int, error_type: str) -> np.ndarray:
    clip = np.repeat(_base_pose()[None, :, :], t, axis=0)
    phase = np.sin(np.linspace(0.0, math.pi, t, dtype=np.float32))

    for i, p in enumerate(phase):
        depth = 0.16 * p
        knee_bend = 0.08 * p
        clip[i, [23, 24], 1] += 0.06 * p
        clip[i, [25, 26], 1] += depth
        clip[i, [27, 28], 1] += 0.02 * p
        clip[i, 25, 0] += knee_bend
        clip[i, 26, 0] -= knee_bend

        if error_type == "knee_valgus":
            clip[i, 25, 0] += 0.11 * p
            clip[i, 26, 0] -= 0.11 * p
        elif error_type == "insufficient_depth":
            clip[i, [23, 24, 25, 26], 1] -= 0.08 * p
        elif error_type == "back_rounding":
            clip[i, [11, 12], 0] += 0.08 * p
            clip[i, [11, 12], 1] += 0.05 * p

    clip[:, :, :2] += np.random.normal(0.0, 0.012, size=(t, 33, 2)).astype(np.float32)
    clip[:, :, 2] = np.random.normal(0.0, 0.03, size=(t, 33)).astype(np.float32)
    clip[:, :, 3] = np.clip(
        np.random.normal(0.92, 0.04, size=(t, 33)).astype(np.float32), 0.0, 1.0
    )
    return clip.astype(np.float32)


def _make_curl_clip(t: int, error_type: str) -> np.ndarray:
    clip = np.repeat(_base_pose()[None, :, :], t, axis=0)
    phase = np.sin(np.linspace(0.0, math.pi, t, dtype=np.float32))

    for i, p in enumerate(phase):
        clip[i, 14, :2] = (0.63, 0.42)
        clip[i, 16, 0] = 0.64 - 0.12 * p
        clip[i, 16, 1] = 0.58 - 0.20 * p

        if error_type == "swing":
            sway = 0.07 * np.sin(2.0 * math.pi * i / max(t - 1, 1))
            clip[i, [11, 12, 23, 24], 0] += sway
            clip[i, [11, 12], 1] += 0.03 * p
        elif error_type == "partial_rom":
            clip[i, 16, 1] += 0.10 * p
            clip[i, 16, 0] += 0.06 * p
        elif error_type == "elbow_flare":
            clip[i, 14, 0] += 0.11 * p
            clip[i, 16, 0] += 0.08 * p

    clip[:, :, :2] += np.random.normal(0.0, 0.012, size=(t, 33, 2)).astype(np.float32)
    clip[:, :, 2] = np.random.normal(0.0, 0.03, size=(t, 33)).astype(np.float32)
    clip[:, :, 3] = np.clip(
        np.random.normal(0.92, 0.04, size=(t, 33)).astype(np.float32), 0.0, 1.0
    )
    return clip.astype(np.float32)


def generate_synthetic_data(root: Path = DATA_DIR) -> None:
    if root.exists():
        shutil.rmtree(root)
    (root / "squat").mkdir(parents=True, exist_ok=True)
    (root / "curl").mkdir(parents=True, exist_ok=True)

    rows = []
    subjects = [f"subject_{i:02d}" for i in range(1, 9)]
    exercise_errors = {
        "squat": ["none"] * 80
        + ["knee_valgus"] * 27
        + ["insufficient_depth"] * 27
        + ["back_rounding"] * 26,
        "curl": ["none"] * 80 + ["swing"] * 27 + ["partial_rom"] * 27 + ["elbow_flare"] * 26,
    }

    for exercise, errors in exercise_errors.items():
        random.shuffle(errors)
        for idx, error_type in enumerate(errors, start=1):
            t = random.randint(20, 60)
            label = "good" if error_type == "none" else "bad"
            clip_id = f"{exercise}_{idx:04d}"
            file_rel = f"{exercise}/clip_{idx:04d}.npy"
            subject_id = subjects[(idx - 1) % len(subjects)]
            clip = _make_squat_clip(t, error_type) if exercise == "squat" else _make_curl_clip(t, error_type)

            np.save(root / file_rel, clip.astype(np.float32))
            rows.append(
                {
                    "clip_id": clip_id,
                    "file": file_rel,
                    "exercise": exercise,
                    "view": "threequarter" if exercise == "squat" else "front",
                    "label": label,
                    "error_type": error_type,
                    "subject_id": subject_id,
                    "fps": 30,
                    "num_frames": t,
                    "notes": "synthetic",
                }
            )

    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(root / "manifest.csv", index=False)


# Preprocessing recipe
def _angle_at_b(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _angle_between(v: np.ndarray, axis: np.ndarray) -> float:
    denom = np.linalg.norm(v) * np.linalg.norm(axis)
    if denom < 1e-8:
        return 0.0
    cosang = np.clip(np.dot(v, axis) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _resample_features(features: np.ndarray, w: int = W) -> np.ndarray:
    if features.shape[0] == w:
        return features.astype(np.float32)
    if features.shape[0] == 1:
        return np.repeat(features, w, axis=0).astype(np.float32)

    src_x = np.linspace(0.0, 1.0, features.shape[0], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    out = np.empty((w, features.shape[1]), dtype=np.float32)
    for j in range(features.shape[1]):
        out[:, j] = np.interp(dst_x, src_x, features[:, j]).astype(np.float32)
    return out


def preprocess_rep(raw: np.ndarray, exercise: str) -> np.ndarray:
    if exercise not in {"squat", "curl"}:
        raise ValueError("exercise must be 'squat' or 'curl'")
    if raw.ndim != 3 or raw.shape[1:] != (33, 4):
        raise ValueError("raw must have shape [T, 33, 4]")

    required = [11, 12, 23, 24]
    if exercise == "squat":
        required += [25, 26, 27, 28]
    else:
        required += [14, 16]

    frames = []
    for frame in raw.astype(np.float32, copy=False):
        xy = frame[:, :2]
        if not np.isfinite(xy[required]).all():
            continue

        hip_mid = (xy[23] + xy[24]) / 2.0
        sh_mid = (xy[11] + xy[12]) / 2.0
        scale = np.linalg.norm(sh_mid - hip_mid)
        if scale < 1e-4:
            continue

        norm_xy = (xy - hip_mid) / scale
        sh_mid_norm = (norm_xy[11] + norm_xy[12]) / 2.0

        if exercise == "squat":
            feature = [
                norm_xy[11, 0],
                norm_xy[11, 1],
                norm_xy[12, 0],
                norm_xy[12, 1],
                norm_xy[23, 0],
                norm_xy[23, 1],
                norm_xy[24, 0],
                norm_xy[24, 1],
                norm_xy[25, 0],
                norm_xy[25, 1],
                norm_xy[26, 0],
                norm_xy[26, 1],
                norm_xy[27, 0],
                norm_xy[27, 1],
                norm_xy[28, 0],
                norm_xy[28, 1],
                _angle_at_b(norm_xy[23], norm_xy[25], norm_xy[27]) / 180.0,
                _angle_at_b(norm_xy[24], norm_xy[26], norm_xy[28]) / 180.0,
                _angle_between(sh_mid_norm, np.array([0.0, -1.0], dtype=np.float32)) / 180.0,
                norm_xy[25, 0] - norm_xy[27, 0],
                norm_xy[26, 0] - norm_xy[28, 0],
            ]
        else:
            feature = [
                norm_xy[12, 0],
                norm_xy[12, 1],
                norm_xy[14, 0],
                norm_xy[14, 1],
                norm_xy[16, 0],
                norm_xy[16, 1],
                norm_xy[11, 0],
                norm_xy[11, 1],
                _angle_at_b(norm_xy[12], norm_xy[14], norm_xy[16]) / 180.0,
                _angle_between(norm_xy[12] - norm_xy[11], np.array([1.0, 0.0], dtype=np.float32))
                / 180.0,
                norm_xy[14, 0] - norm_xy[12, 0],
            ]
        frames.append(feature)

    if not frames:
        f = len(SQUAT_FEATURE_ORDER) if exercise == "squat" else len(CURL_FEATURE_ORDER)
        raise ValueError(f"no valid frames available for {exercise} preprocessing")

    features = np.asarray(frames, dtype=np.float32)
    if not np.isfinite(features).all():
        features = features[np.isfinite(features).all(axis=1)]
    if features.shape[0] == 0:
        f = len(SQUAT_FEATURE_ORDER) if exercise == "squat" else len(CURL_FEATURE_ORDER)
        raise ValueError(f"no finite feature frames available for {exercise} preprocessing with F={f}")
    return _resample_features(features, W).astype(np.float32)


# Dataset loader and subject-held-out split
def load_dataset(root: Path, exercise: str):
    manifest = pd.read_csv(root / "manifest.csv")
    manifest = manifest[manifest["exercise"] == exercise].copy()

    xs, ys, subjects, clip_ids = [], [], [], []
    for _, row in manifest.iterrows():
        raw = np.load(root / row["file"]).astype(np.float32)
        if raw.ndim != 3 or raw.shape[1:] != (33, 4):
            raise ValueError(f"{row['file']} has invalid shape {raw.shape}")

        finite_frame_mask = np.isfinite(raw[:, :, :2]).all(axis=(1, 2))
        raw = raw[finite_frame_mask]
        if raw.shape[0] == 0:
            continue

        try:
            x = preprocess_rep(raw, exercise)
        except ValueError:
            continue
        y = ERROR_TO_CLASS[exercise][row["error_type"]]
        xs.append(x)
        ys.append(y)
        subjects.append(row["subject_id"])
        clip_ids.append(row["clip_id"])

    x_np = np.stack(xs).astype(np.float32)
    y_np = np.asarray(ys, dtype=np.int64)
    return torch.from_numpy(x_np), torch.from_numpy(y_np), np.asarray(subjects), clip_ids


def subject_held_out_split(subjects: np.ndarray, holdout_count: int = 2):
    unique_subjects = np.array(sorted(set(subjects.tolist())))
    val_subjects = set(unique_subjects[-holdout_count:].tolist())
    val_mask = np.array([s in val_subjects for s in subjects], dtype=bool)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx, sorted(val_subjects)


# Model definition
class FormScorer(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=4):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


# Training loop
def _class_weights(y_train: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(y_train, minlength=num_classes).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    return weights.to(device)


def train_model(exercise: str, x: torch.Tensor, y: torch.Tensor, subjects: np.ndarray) -> FormScorer:
    f = x.shape[-1]
    num_classes = len(CLASS_LABELS[exercise])
    train_idx, val_idx, val_subjects = subject_held_out_split(subjects)
    print(f"{exercise}: train={len(train_idx)} val={len(val_idx)} val_subjects={val_subjects}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FormScorer(in_dim=f, hidden=64, num_classes=num_classes).to(device)

    x_train = x[train_idx].to(device)
    y_train = y[train_idx].to(device)
    x_val = x[val_idx].to(device)
    y_val = y[val_idx].to(device)

    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_train.cpu(), num_classes, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, 31):
        model.train()
        perm = torch.randperm(x_train.shape[0], device=device)
        total_loss = 0.0
        total_count = 0

        for start in range(0, x_train.shape[0], 32):
            idx = perm[start : start + 32]
            xb = x_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item()) * xb.shape[0]
            total_count += xb.shape[0]

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_true = y_val.cpu().numpy()
            val_acc = accuracy_score(val_true, val_pred)

        avg_loss = total_loss / max(total_count, 1)
        print(f"{exercise} epoch {epoch:02d}/30 loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    with torch.no_grad():
        final_pred = model(x_val).argmax(dim=1).cpu().numpy()
        final_true = y_val.cpu().numpy()
    print(f"{exercise} confusion_matrix:")
    print(confusion_matrix(final_true, final_pred, labels=list(range(num_classes))))
    return model.cpu().eval()


# Core ML export and model cards
def export_coreml(model: FormScorer, exercise: str, f: int) -> None:
    package_name = "SquatFormScorer.mlpackage" if exercise == "squat" else "CurlFormScorer.mlpackage"
    example = torch.rand(1, W, f, dtype=torch.float32)
    traced = torch.jit.trace(model, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="keypoint_window", shape=(1, W, f))],
        outputs=[ct.TensorType(name="form_logits")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.save(package_name)


def write_model_card(exercise: str, f: int) -> None:
    card = {
        "exercise": exercise,
        "W": W,
        "F": f,
        "feature_order": SQUAT_FEATURE_ORDER if exercise == "squat" else CURL_FEATURE_ORDER,
        "class_labels": CLASS_LABELS[exercise],
        "normalization": "hip-center origin, torso-length scale (see contracts sec 7)",
        "version": "v1",
        "built": date.today().isoformat(),
    }
    out = "SquatFormScorer_model_card.json" if exercise == "squat" else "CurlFormScorer_model_card.json"
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(card, fp, indent=2)
        fp.write("\n")


# Golden test artifact
def write_golden_test(root: Path = DATA_DIR) -> None:
    manifest = pd.read_csv(root / "manifest.csv")
    squat_rows = manifest[manifest["exercise"] == "squat"].head(3)
    reps = []
    for _, row in squat_rows.iterrows():
        raw = np.load(root / row["file"]).astype(np.float32)
        reps.append(preprocess_rep(raw, "squat"))
    np.save("golden_test.npy", np.stack(reps).astype(np.float32))


# Main pipeline
def main() -> None:
    set_all_seeds(SEED)
    generate_synthetic_data(DATA_DIR)

    squat_probe = preprocess_rep(np.load(DATA_DIR / "squat/clip_0001.npy"), "squat")
    curl_probe = preprocess_rep(np.load(DATA_DIR / "curl/clip_0001.npy"), "curl")
    assert squat_probe.shape == (W, len(SQUAT_FEATURE_ORDER))
    assert curl_probe.shape == (W, len(CURL_FEATURE_ORDER))
    assert np.isfinite(squat_probe).all()
    assert np.isfinite(curl_probe).all()

    write_golden_test(DATA_DIR)

    for exercise in ("squat", "curl"):
        x, y, subjects, _ = load_dataset(DATA_DIR, exercise)
        model = train_model(exercise, x, y, subjects)
        f = x.shape[-1]
        export_coreml(model, exercise, f)
        write_model_card(exercise, f)


if __name__ == "__main__":
    main()
