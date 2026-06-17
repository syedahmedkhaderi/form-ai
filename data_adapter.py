#!/usr/bin/env python3
"""
data_adapter.py — builds formai_data/ entirely from real CSV pose data.

Run this BEFORE formai_pipeline.py.

Sources (already downloaded to /tmp/):
  /tmp/squat_train.csv  4 160 frames  label: up / down (stage labels)
  /tmp/squat_test.csv     853 frames  label: up / down
  /tmp/curl_train.csv  15 372 frames  label: C=Correct / L=Lean-back
  /tmp/curl_test.csv      604 frames  label: C=Correct / L=Lean-back

What this script produces
──────────────────────────
formai_data/squat/clip_NNNN.npy  [T, 33, 4] float32 — all classes
  good          ← real reps segmented by up→down→up transitions
  knee_valgus   ← real good rep + knees shifted inward (calibrated to real geometry)
  insufficient_depth ← real good rep + squat depth reduced by ~55%
  back_rounding ← real good rep + shoulders shifted forward relative to hips

formai_data/curl/clip_NNNN.npy  [T, 33, 4] float32 — 2 classes
  good          ← real reps (C label)
  swing         ← real lean-back reps (L label)

Why augmentation not pure synthetic for squat
  The old synthetic generator had completely wrong geometry: its knees stayed BELOW
  the hips in image space, while real squats show knees ABOVE the hip joint in the
  "down" phase (hip drops past knee level). The model trained on fake geometry scores
  randomly on a real person. This approach keeps the base pose real and applies
  clinically-meaningful perturbations so all four squat classes share the same
  real-human coordinate distributions.

Key stats measured from the CSVs (used to calibrate perturbation magnitudes)
  Up phase  : hip_y≈0.44  l_valgus≈-0.04  r_valgus≈+0.04  (torso≈0.226)
  Down phase: hip_y≈0.63  l_valgus≈+0.09  r_valgus≈-0.09  (torso≈0.192)
  Hip travel: ~0.19 image units (standing → squat depth)
"""

import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

DATA_DIR = Path("formai_data")

# MediaPipe landmark index for joint names in the squat CSV
SQUAT_COL_TO_LM = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# MediaPipe landmark index for joint names in the curl CSV
CURL_COL_TO_LM = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

MANIFEST_COLUMNS = [
    "clip_id", "file", "exercise", "view", "label",
    "error_type", "subject_id", "split", "fps", "num_frames", "notes",
]

CURL_SOURCES = [
    (
        "train",
        Path("/tmp/curl_train.csv"),
        "https://raw.githubusercontent.com/NgoQuocBao1010/Exercise-Correction/main/core/bicep_model/train.csv",
    ),
    (
        "test",
        Path("/tmp/curl_test.csv"),
        "https://raw.githubusercontent.com/NgoQuocBao1010/Exercise-Correction/main/core/bicep_model/test.csv",
    ),
]


# ── shared helpers ────────────────────────────────────────────────────────────

def df_to_landmarks(df, col_to_lm):
    """DataFrame → [T, 33, 4] float32 using the joint-name → LM-index map."""
    T = len(df)
    arr = np.zeros((T, 33, 4), dtype=np.float32)
    for name, idx in col_to_lm.items():
        for j, sfx in enumerate(["x", "y", "z", "v"]):
            col = f"{name}_{sfx}"
            arr[:, idx, j] = df[col].values.astype(np.float32) if col in df.columns else (
                1.0 if sfx == "v" else 0.0
            )
    return arr


def angle_at_b(a, b, c):
    """Angle in degrees at vertex b formed by a–b–c."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def moving_avg(arr, k=7):
    return np.convolve(arr, np.ones(k) / k, mode="same")


def find_peaks(signal, min_val, min_gap):
    """Local maxima above min_val spaced at least min_gap frames apart."""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] < min_val:
            continue
        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_gap:
                peaks.append(i)
            elif signal[i] > signal[peaks[-1]]:
                peaks[-1] = i
    return peaks


def find_video_breaks(df, threshold=0.05):
    """Indices where nose position jumps > threshold → new video file boundary."""
    dx = np.diff(df["nose_x"].values)
    dy = np.diff(df["nose_y"].values)
    breaks = [0] + list(np.where(np.hypot(dx, dy) > threshold)[0] + 1) + [len(df)]
    return breaks


def dominant_label(labels, min_fraction=0.9):
    """Return the majority label if it clearly dominates, else None."""
    if len(labels) == 0:
        return None
    counts = pd.Series(labels).value_counts(normalize=True)
    if counts.empty or float(counts.iloc[0]) < min_fraction:
        return None
    return str(counts.index[0])


def load_csv_source(local_path, upstream_url):
    """Load a CSV from /tmp if present, otherwise fall back to the upstream raw URL."""
    if local_path.exists():
        return pd.read_csv(local_path), str(local_path)
    return pd.read_csv(upstream_url), upstream_url


def jitter(frames, std=0.006):
    """Add small positional noise so all clips are slightly different."""
    T = len(frames)
    out = frames.copy()
    out[:, :, :2] += RNG.standard_normal((T, 33, 2)).astype(np.float32) * std
    return out


# ── squat: real data + augmented error examples ───────────────────────────────

def _perturb_valgus(frames):
    """
    Knee valgus: knees cave inward relative to ankles.

    In image coordinates (y↓), the person's anatomical LEFT side appears on the
    RIGHT of the frame (larger x) and vice versa.

    Normal "down" phase (measured from CSV):
      l_valgus ≈ +0.09   (left knee is outside its ankle — correct)
      r_valgus ≈ -0.09   (right knee is outside its ankle — correct)

    Valgus error:
      left knee (LM 25) shifts LEFT  in image (x decreases) → medial collapse
      right knee (LM 26) shifts RIGHT in image (x increases) → medial collapse

    Magnitude calibrated so the valgus signal clearly exceeds normal range.
    """
    T = len(frames)
    out = frames.copy()
    phase = np.sin(np.linspace(0.0, np.pi, T, dtype=np.float32))
    for i, p in enumerate(phase):
        out[i, 25, 0] -= 0.08 * p   # left knee moves left (inward)
        out[i, 26, 0] += 0.08 * p   # right knee moves right (inward)
    return jitter(out)


def _perturb_insufficient_depth(frames):
    """
    Insufficient depth: squat doesn't go deep enough.

    The real rep's hip_y travels ≈ 0.19 from standing to depth (measured from CSV).
    We reduce that travel by ~55% — enough to keep the knee angle shallow.
    """
    T = len(frames)
    out = frames.copy()
    phase = np.sin(np.linspace(0.0, np.pi, T, dtype=np.float32))
    # 0.10 ≈ 55% of 0.19 hip-travel in image coords
    depth_cancel = 0.10
    for i, p in enumerate(phase):
        out[i, 23, 1] -= depth_cancel * p   # left hip back up
        out[i, 24, 1] -= depth_cancel * p   # right hip back up
        out[i, 25, 1] -= depth_cancel * 0.7 * p  # left knee (moves less)
        out[i, 26, 1] -= depth_cancel * 0.7 * p  # right knee
    return jitter(out)


def _perturb_back_rounding(frames):
    """
    Back rounding / forward torso lean.

    Shoulders shift down (toward hips, y increases) as the squat deepens.
    This increases the torso_lean feature (shoulder–hip angle from vertical).
    Magnitude chosen so torso_lean exceeds the normal range clearly.
    """
    T = len(frames)
    out = frames.copy()
    phase = np.sin(np.linspace(0.0, np.pi, T, dtype=np.float32))
    for i, p in enumerate(phase):
        out[i, 11, 1] += 0.045 * p  # left shoulder down toward hip
        out[i, 12, 1] += 0.045 * p  # right shoulder down toward hip
    return jitter(out)


SQUAT_AUGMENTERS = {
    "knee_valgus": _perturb_valgus,
    "insufficient_depth": _perturb_insufficient_depth,
    "back_rounding": _perturb_back_rounding,
}


def segment_squat_reps(df, min_frames=15, max_frames=200):
    """
    Return list of [T, 33, 4] arrays by detecting up→down→up transitions.
    All returned reps are labeled 'good' — error variants are created by augmentation.

    No video-break splitting: the squat CSV is one continuous recording and label
    transitions correctly identify rep boundaries.
    """
    labels_col = df["label"].values
    lm = df_to_landmarks(df, SQUAT_COL_TO_LM)

    reps = []
    n = len(labels_col)
    i = 0

    while i < n:
        # Advance to start of 'up' phase
        while i < n and labels_col[i] != "up":
            i += 1
        if i >= n:
            break
        start = i

        # Collect 'up' phase
        while i < n and labels_col[i] == "up":
            i += 1

        # 'down' phase must follow immediately
        if i >= n or labels_col[i] != "down":
            continue
        while i < n and labels_col[i] == "down":
            i += 1

        end = i
        length = end - start
        if min_frames <= length <= max_frames:
            reps.append(lm[start:end].copy())

    return reps


def generate_real_squat(root, rows):
    """
    Process squat CSVs → real good reps + augmented error reps.
    Appends manifest rows in-place.
    """
    csv_paths = [Path("/tmp/squat_train.csv"), Path("/tmp/squat_test.csv")]
    available = [p for p in csv_paths if p.exists()]
    if not available:
        print("WARNING: no squat CSVs found — skipping real squat data")
        return

    squat_dir = root / "squat"
    squat_dir.mkdir(parents=True, exist_ok=True)
    subjects = [f"subject_{i:02d}" for i in range(1, 9)]
    clip_counter = 1

    all_reps = []
    for csv_path in available:
        print(f"  Loading {csv_path.name} ...")
        df = pd.read_csv(csv_path)
        # Process the full CSV as one sequence — the label transitions (up↔down)
        # correctly identify rep boundaries without needing video-break detection.
        all_reps.extend(segment_squat_reps(df))

    if not all_reps:
        print("WARNING: no squat reps segmented — check CSV contents")
        return

    print(f"  Segmented {len(all_reps)} real good squat reps")
    print(f"  Creating 4× clips (1 good + 3 augmented error per rep) ...")

    for rep_idx, good_rep in enumerate(all_reps):
        subject_id = subjects[rep_idx % len(subjects)]
        T = len(good_rep)

        # Save the real good rep
        for error_type, aug_fn in [("none", jitter)] + list(SQUAT_AUGMENTERS.items()):
            clip = aug_fn(good_rep)
            file_rel = f"squat/clip_{clip_counter:04d}.npy"
            np.save(root / file_rel, clip.astype(np.float32))
            rows.append({
                "clip_id": f"squat_{clip_counter:04d}",
                "file": file_rel,
                "exercise": "squat",
                "view": "front",
                "label": "good" if error_type == "none" else "bad",
                "error_type": error_type,
                "subject_id": subject_id,
                "split": "train",
                "fps": 30,
                "num_frames": T,
                "notes": "real" if error_type == "none" else "real+augmented",
            })
            clip_counter += 1

    n_reps = len(all_reps)
    print(f"  Wrote {clip_counter - 1} squat clips ({n_reps} good + "
          f"{n_reps} valgus + {n_reps} depth + {n_reps} rounding)")


# ── curl: real data (C=good, L=swing) ────────────────────────────────────────

def append_curl_clip(rows, curl_dir, clip_counter, clip, dominant, source_id, split_name, source_ref):
    """Persist one real curl rep and append its manifest row."""
    quality = "good" if dominant == "C" else "bad"
    error_type = "none" if dominant == "C" else "swing"
    file_rel = f"curl/clip_{clip_counter:04d}.npy"
    np.save(curl_dir / f"clip_{clip_counter:04d}.npy", clip.astype(np.float32))
    rows.append({
        "clip_id": f"curl_{clip_counter:04d}",
        "file": file_rel,
        "exercise": "curl",
        "view": "front",
        "label": quality,
        "error_type": error_type,
        "subject_id": source_id,
        "split": split_name,
        "fps": 30,
        "num_frames": len(clip),
        "notes": f"real — NgoQuocBao1010/Exercise-Correction bicep curl ({source_ref})",
    })
    return clip_counter + 1


def process_curl_video_segment(df_seg, lm_seg, rows, clip_counter, curl_dir,
                               source_id, split_name, source_ref,
                               min_frames=15, max_frames=150, purity=0.9):
    """Turn one source video segment into one or more high-purity rep clips."""
    labels_col = df_seg["label"].values

    # Some upstream test segments are already isolated single reps.
    dominant = dominant_label(labels_col, min_fraction=purity)
    if dominant is not None and min_frames <= len(df_seg) <= 90:
        return append_curl_clip(
            rows, curl_dir, clip_counter, lm_seg, dominant, source_id, split_name, source_ref
        )

    sh = lm_seg[:, 12, :2]
    el = lm_seg[:, 14, :2]
    wr = lm_seg[:, 16, :2]
    angles = np.array([angle_at_b(sh[i], el[i], wr[i]) for i in range(len(df_seg))])
    peaks = find_peaks(moving_avg(angles), min_val=120.0, min_gap=15)

    if len(peaks) < 2:
        return clip_counter

    for k in range(len(peaks) - 1):
        start, end = peaks[k], peaks[k + 1]
        if not (min_frames <= end - start <= max_frames):
            continue
        dominant = dominant_label(labels_col[start:end], min_fraction=purity)
        if dominant is None:
            continue
        clip_counter = append_curl_clip(
            rows,
            curl_dir,
            clip_counter,
            lm_seg[start:end],
            dominant,
            source_id,
            split_name,
            source_ref,
        )

    return clip_counter


def generate_real_curl(root, rows):
    """Process curl CSVs and write per-rep clips using the upstream train/test split."""
    curl_dir = root / "curl"
    curl_dir.mkdir(parents=True, exist_ok=True)
    clip_counter = 1
    loaded_any = False

    for split_name, local_path, upstream_url in CURL_SOURCES:
        try:
            df, source_ref = load_csv_source(local_path, upstream_url)
        except Exception as exc:
            print(f"  WARNING: failed to load curl {split_name} CSV: {exc}")
            continue

        loaded_any = True
        print(f"  Loading curl {split_name} from {source_ref} ...")
        breaks = find_video_breaks(df)
        for seg_idx, (seg_start, seg_end) in enumerate(zip(breaks[:-1], breaks[1:])):
            if seg_end - seg_start < 15:
                continue
            df_seg = df.iloc[seg_start:seg_end].reset_index(drop=True)
            lm_seg = df_to_landmarks(df_seg, CURL_COL_TO_LM)
            source_id = f"{split_name}_seg_{seg_idx:04d}"
            clip_counter = process_curl_video_segment(
                df_seg,
                lm_seg,
                rows,
                clip_counter,
                curl_dir,
                source_id,
                split_name,
                source_ref,
            )

    if not loaded_any:
        print("  WARNING: no curl CSVs found — skipping")
        return


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("data_adapter.py — building formai_data/ from real CSV data")
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)

    rows = []

    print("Squat (real data + augmented error variants):")
    generate_real_squat(DATA_DIR, rows)

    print("Curl (real data, 2-class C/L):")
    generate_real_curl(DATA_DIR, rows)

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest.to_csv(DATA_DIR / "manifest.csv", index=False)

    for ex in ["squat", "curl"]:
        sub = manifest[manifest["exercise"] == ex]
        parts = [f"{v}: {n}" for v, n in sub["error_type"].value_counts().items()]
        print(f"  {ex}: {len(sub)} clips — {', '.join(parts)}")

    print(f"\nTotal: {len(manifest)} clips → formai_data/manifest.csv")


if __name__ == "__main__":
    main()
