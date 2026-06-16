#!/usr/bin/env python3
"""Adapter for NgoQuocBao1010/Exercise-Correction models.

The upstream project works frame-by-frame on MediaPipe Pose landmarks and ships
pickled sklearn models plus deterministic angle/ratio checks. FormAI's current
interface works rep-by-rep on raw MediaPipe tensors shaped [T, 33, 4].

This module bridges those two worlds without training new weights.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
UPSTREAM_ROOT = ROOT / "third_party" / "Exercise-Correction"
UPSTREAM_STATIC_MODELS = UPSTREAM_ROOT / "web" / "server" / "static" / "model"
POSTURE_ROOT = ROOT / "third_party" / "Posture"
POSTURE_MODEL_DIR = POSTURE_ROOT / "working_model_1"

W = 32

SQUAT_LABELS = ["good", "knee_valgus", "insufficient_depth", "back_rounding"]
CURL_LABELS = ["good", "swing", "partial_rom", "elbow_flare"]
POSTURE_OUTPUT_LABELS = ["correct", "knee", "hip", "rounded_back", "depth"]

SQUAT_IMPORTANT_LANDMARKS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]

CURL_IMPORTANT_LANDMARKS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "LEFT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
]

LANDMARK_INDEX = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}


@dataclass(frozen=True)
class FormAIPrediction:
    exercise: str
    label: str
    class_labels: list[str]
    logits: list[float]
    confidence: float
    issues: list[str]
    source: str
    model_available: bool
    model_load_error: str | None = None


def _headers(landmarks: Iterable[str]) -> list[str]:
    cols = []
    for lm in landmarks:
        name = lm.lower()
        cols.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_v"])
    return cols


def _validate_raw(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (33, 4):
        raise ValueError("raw landmarks must have shape [T, 33, 4]")

    finite = np.isfinite(raw).all(axis=(1, 2))
    raw = raw[finite]
    if raw.shape[0] == 0:
        raise ValueError("raw landmarks contain no finite frames")
    return raw


def _frame_to_row(frame: np.ndarray, important_landmarks: list[str]) -> list[float]:
    row: list[float] = []
    for name in important_landmarks:
        row.extend(frame[LANDMARK_INDEX[name], :4].astype(float).tolist())
    return row


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(
        a[1] - b[1], a[0] - b[0]
    )
    degrees = abs(radians * 180.0 / math.pi)
    return degrees if degrees <= 180.0 else 360.0 - degrees


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    a_xy = np.asarray(a[:2], dtype=np.float32)
    b_xy = np.asarray(b[:2], dtype=np.float32)
    return float(np.linalg.norm(a_xy - b_xy))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator < 1e-6:
        return None
    return round(numerator / denominator, 1)


def _safe_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-8:
        return float("nan")
    cos_theta = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(math.acos(cos_theta))


def _load_pickle(path: Path):
    if not path.exists():
        return None, f"missing model file: {path}"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with path.open("rb") as fp:
                return pickle.load(fp), None
    except Exception as exc:  # sklearn pickles are version-fragile by design.
        return None, f"{type(exc).__name__}: {exc}"


def _as_logits(labels: list[str], winner: str, confidence: float) -> list[float]:
    floor = -2.0
    logits = [floor] * len(labels)
    logits[labels.index(winner)] = max(0.0, min(1.0, confidence)) * 6.0
    return logits


def _prediction(
    exercise: str,
    labels: list[str],
    winner: str,
    issues: list[str],
    source: str,
    model_available: bool,
    model_load_error: str | None,
    confidence: float = 0.9,
) -> FormAIPrediction:
    return FormAIPrediction(
        exercise=exercise,
        label=winner,
        class_labels=labels,
        logits=_as_logits(labels, winner, confidence),
        confidence=confidence,
        issues=issues,
        source=source,
        model_available=model_available,
        model_load_error=model_load_error,
    )


class ExerciseCorrectionAdapter:
    """Rep-level inference facade backed by the upstream Exercise-Correction repo."""

    def __init__(self, model_dir: Path = UPSTREAM_STATIC_MODELS) -> None:
        self.model_dir = Path(model_dir)
        self.squat_model, self.squat_model_error = _load_pickle(self.model_dir / "squat_model.pkl")
        self.curl_model, self.curl_model_error = _load_pickle(self.model_dir / "bicep_curl_model.pkl")
        self.curl_scaler, self.curl_scaler_error = _load_pickle(
            self.model_dir / "bicep_curl_input_scaler.pkl"
        )
        self.posture_model = None
        self.posture_model_error: str | None = None
        self._posture_model_loaded = False

    def predict_squat(self, raw: np.ndarray) -> FormAIPrediction:
        raw = _validate_raw(raw)
        posture_features = _posture_squat_features(raw)
        posture_scores = self._predict_posture_squat_scores(posture_features)
        if posture_scores is not None:
            label, issues, confidence = _posture_scores_to_formai(posture_scores)
            return _prediction(
                "squat",
                SQUAT_LABELS,
                label,
                issues,
                "Posture TensorFlow squat model + FormAI label mapping",
                self.posture_model is not None,
                self.posture_model_error,
                confidence=confidence,
            )

        issues: list[str] = []
        stage_by_model = self._predict_squat_stages(raw)
        if self._has_squat_knee_issue(raw, stage_by_model):
            issues.append("Posture model unavailable; knee placement outside threshold")
            return _prediction(
                "squat",
                SQUAT_LABELS,
                "knee_valgus",
                issues,
                self._source_name("squat", stage_by_model is not None),
                self.squat_model is not None,
                self.squat_model_error,
            )

        if self._has_insufficient_squat_depth(raw):
            issues.append("Posture model unavailable; minimum knee angle did not pass depth threshold")
            return _prediction(
                "squat",
                SQUAT_LABELS,
                "insufficient_depth",
                issues,
                self._source_name("squat", stage_by_model is not None),
                self.squat_model is not None,
                self.squat_model_error,
                confidence=0.85,
            )

        if self._has_back_rounding(raw):
            issues.append("Posture model unavailable; torso lean exceeded conservative threshold")
            return _prediction(
                "squat",
                SQUAT_LABELS,
                "back_rounding",
                issues,
                self._source_name("squat", stage_by_model is not None),
                self.squat_model is not None,
                self.squat_model_error,
                confidence=0.78,
            )

        return _prediction(
            "squat",
            SQUAT_LABELS,
            "good",
            issues,
            self._source_name("squat", stage_by_model is not None),
            self.squat_model is not None,
            self.squat_model_error,
        )

    def predict_curl(self, raw: np.ndarray, side: str = "right") -> FormAIPrediction:
        raw = _validate_raw(raw)
        side = side.lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")

        issues: list[str] = []
        lean_back = self._predict_curl_lean_back(raw)
        curl_model_used = lean_back is not None
        if lean_back:
            issues.append("upstream bicep posture model detected lean-back")
            return _prediction(
                "curl",
                CURL_LABELS,
                "swing",
                issues,
                self._source_name("curl", curl_model_used),
                self.curl_model is not None and self.curl_scaler is not None,
                self.curl_model_error or self.curl_scaler_error,
            )

        elbow_angles = self._curl_elbow_angles(raw, side)
        if elbow_angles.size and float(elbow_angles.min()) >= 60.0:
            issues.append("peak contraction angle stayed above upstream threshold")
            return _prediction(
                "curl",
                CURL_LABELS,
                "partial_rom",
                issues,
                self._source_name("curl", curl_model_used),
                self.curl_model is not None and self.curl_scaler is not None,
                self.curl_model_error or self.curl_scaler_error,
                confidence=0.86,
            )

        upper_arm_angles = self._curl_upper_arm_angles(raw, side)
        if upper_arm_angles.size and float(np.nanmax(upper_arm_angles)) > 40.0:
            issues.append("upper arm angle exceeded upstream loose-arm threshold")
            return _prediction(
                "curl",
                CURL_LABELS,
                "elbow_flare",
                issues,
                self._source_name("curl", curl_model_used),
                self.curl_model is not None and self.curl_scaler is not None,
                self.curl_model_error or self.curl_scaler_error,
                confidence=0.84,
            )

        return _prediction(
            "curl",
            CURL_LABELS,
            "good",
            issues,
            self._source_name("curl", curl_model_used),
            self.curl_model is not None and self.curl_scaler is not None,
            self.curl_model_error or self.curl_scaler_error,
        )

    def predict(self, raw: np.ndarray, exercise: str, side: str = "right") -> FormAIPrediction:
        if exercise == "squat":
            return self.predict_squat(raw)
        if exercise in {"curl", "bicep_curl"}:
            return self.predict_curl(raw, side=side)
        raise ValueError("exercise must be 'squat', 'curl', or 'bicep_curl'")

    def write_metadata(self, output_path: Path = ROOT / "exercise_correction_adapter_card.json") -> None:
        self._load_posture_model()
        payload = {
            "source_repo": "https://github.com/NgoQuocBao1010/Exercise-Correction",
            "source_path": str(UPSTREAM_ROOT.relative_to(ROOT)),
            "interface": "raw MediaPipe Pose landmarks [T,33,4] -> FormAIPrediction logits/label",
            "class_labels": {"squat": SQUAT_LABELS, "curl": CURL_LABELS},
            "posture_model": {
                "source_repo": "https://github.com/twixupmysleeve/Posture",
                "source_path": str(POSTURE_ROOT.relative_to(ROOT)),
                "output_labels": POSTURE_OUTPUT_LABELS,
                "formai_mapping": {
                    "correct": "good",
                    "knee": "knee_valgus",
                    "hip": "back_rounding",
                    "rounded_back": "back_rounding",
                    "depth": "insufficient_depth",
                },
                "available": self.posture_model is not None,
                "load_error": self.posture_model_error,
            },
            "model_status": {
                "squat_model_available": self.squat_model is not None,
                "squat_model_error": self.squat_model_error,
                "curl_model_available": self.curl_model is not None and self.curl_scaler is not None,
                "curl_model_error": self.curl_model_error or self.curl_scaler_error,
            },
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _load_posture_model(self):
        if self._posture_model_loaded:
            return self.posture_model
        self._posture_model_loaded = True
        if not POSTURE_MODEL_DIR.exists():
            self.posture_model_error = f"missing Posture model directory: {POSTURE_MODEL_DIR}"
            return None
        try:
            import tensorflow as tf  # type: ignore

            self.posture_model = tf.keras.models.load_model(POSTURE_MODEL_DIR)
            self.posture_model_error = None
        except Exception as exc:
            self.posture_model = None
            self.posture_model_error = f"{type(exc).__name__}: {exc}"
        return self.posture_model

    def _predict_posture_squat_scores(self, features: np.ndarray) -> np.ndarray | None:
        model = self._load_posture_model()
        if model is None:
            return None
        if features.size == 0:
            self.posture_model_error = "no finite Posture squat feature frames"
            return None
        try:
            preds = np.asarray(model.predict(features, verbose=0), dtype=np.float32)
        except Exception as exc:
            self.posture_model_error = f"{type(exc).__name__}: {exc}"
            return None
        if preds.ndim != 2 or preds.shape[1] != 5:
            self.posture_model_error = f"unexpected Posture model output shape: {preds.shape}"
            return None
        return np.clip(np.nanmean(preds, axis=0), 0.0, 1.0)

    def _predict_squat_stages(self, raw: np.ndarray) -> list[str] | None:
        if self.squat_model is None:
            return None
        rows = [_frame_to_row(frame, SQUAT_IMPORTANT_LANDMARKS) for frame in raw]
        x = pd.DataFrame(rows, columns=_headers(SQUAT_IMPORTANT_LANDMARKS))
        try:
            return [str(value) for value in self.squat_model.predict(x)]
        except Exception as exc:
            self.squat_model_error = f"{type(exc).__name__}: {exc}"
            self.squat_model = None
            return None

    def _predict_curl_lean_back(self, raw: np.ndarray) -> bool | None:
        if self.curl_model is None or self.curl_scaler is None:
            return None
        rows = [_frame_to_row(frame, CURL_IMPORTANT_LANDMARKS) for frame in raw]
        x = pd.DataFrame(rows, columns=_headers(CURL_IMPORTANT_LANDMARKS))
        try:
            scaled = pd.DataFrame(self.curl_scaler.transform(x))
            predictions = [str(value) for value in self.curl_model.predict(scaled)]
        except Exception as exc:
            self.curl_model_error = f"{type(exc).__name__}: {exc}"
            self.curl_model = None
            return None
        return predictions.count("L") >= max(1, math.ceil(0.2 * len(predictions)))

    def _source_name(self, exercise: str, model_used: bool) -> str:
        if exercise == "squat" and model_used:
            return "Exercise-Correction squat sklearn stage model + upstream threshold rules"
        if exercise == "curl" and model_used:
            return "Exercise-Correction bicep sklearn posture model + upstream threshold rules"
        return "Exercise-Correction upstream threshold rules (pickle unavailable in current sklearn)"

    def _has_squat_knee_issue(self, raw: np.ndarray, stages: list[str] | None) -> bool:
        bad = 0
        seen = 0
        for idx, frame in enumerate(raw):
            stage = stages[idx] if stages else self._squat_stage_from_angles(frame)
            result = _analyze_squat_foot_knee(frame, stage)
            if result["foot_placement"] in {1, 2} or result["knee_placement"] in {1, 2}:
                bad += 1
            if result["foot_placement"] != -1 or result["knee_placement"] != -1:
                seen += 1
        return seen > 0 and bad / seen >= 0.2

    def _has_insufficient_squat_depth(self, raw: np.ndarray) -> bool:
        angles = []
        for frame in raw:
            left = _angle(frame[23, :2], frame[25, :2], frame[27, :2])
            right = _angle(frame[24, :2], frame[26, :2], frame[28, :2])
            angles.append(min(left, right))
        return bool(angles and min(angles) > 105.0)

    def _has_back_rounding(self, raw: np.ndarray) -> bool:
        leans = []
        vertical = np.array([0.0, -1.0], dtype=np.float32)
        for frame in raw:
            hip_mid = (frame[23, :2] + frame[24, :2]) / 2.0
            sh_mid = (frame[11, :2] + frame[12, :2]) / 2.0
            torso = sh_mid - hip_mid
            denom = np.linalg.norm(torso) * np.linalg.norm(vertical)
            if denom < 1e-6:
                continue
            cosang = float(np.clip(np.dot(torso, vertical) / denom, -1.0, 1.0))
            leans.append(math.degrees(math.acos(cosang)))
        return bool(leans and max(leans) > 45.0)

    def _squat_stage_from_angles(self, frame: np.ndarray) -> str:
        left = _angle(frame[23, :2], frame[25, :2], frame[27, :2])
        right = _angle(frame[24, :2], frame[26, :2], frame[28, :2])
        knee = min(left, right)
        if knee < 110.0:
            return "down"
        if knee < 145.0:
            return "middle"
        return "up"

    def _curl_elbow_angles(self, raw: np.ndarray, side: str) -> np.ndarray:
        s, e, w = (11, 13, 15) if side == "left" else (12, 14, 16)
        return np.asarray([_angle(frame[s, :2], frame[e, :2], frame[w, :2]) for frame in raw])

    def _curl_upper_arm_angles(self, raw: np.ndarray, side: str) -> np.ndarray:
        s, e = (11, 13) if side == "left" else (12, 14)
        values = []
        for frame in raw:
            shoulder = frame[s, :2]
            elbow = frame[e, :2]
            projection = np.array([shoulder[0], 1.0], dtype=np.float32)
            values.append(_angle(elbow, shoulder, projection))
        return np.asarray(values, dtype=np.float32)


def _analyze_squat_foot_knee(frame: np.ndarray, stage: str) -> dict[str, int]:
    result = {"foot_placement": -1, "knee_placement": -1}
    if not np.isfinite(frame[[11, 12, 25, 26, 31, 32], :2]).all():
        return result
    if float(np.min(frame[[25, 26, 31, 32], 3])) < 0.6:
        return result

    shoulder_width = _distance(frame[11], frame[12])
    foot_width = _distance(frame[31], frame[32])
    if foot_width < 1e-6:
        return result
    foot_shoulder_ratio = _safe_ratio(foot_width, shoulder_width)
    if foot_shoulder_ratio is None:
        return result

    if 1.2 <= foot_shoulder_ratio <= 2.8:
        result["foot_placement"] = 0
    elif foot_shoulder_ratio < 1.2:
        result["foot_placement"] = 1
    else:
        result["foot_placement"] = 2

    knee_width = _distance(frame[25], frame[26])
    knee_foot_ratio = _safe_ratio(knee_width, foot_width)
    if knee_foot_ratio is None:
        return result

    thresholds = {
        "up": (0.5, 1.0),
        "middle": (0.7, 1.0),
        "down": (0.7, 1.1),
    }.get(stage, (0.7, 1.0))

    lo, hi = thresholds
    if lo <= knee_foot_ratio <= hi:
        result["knee_placement"] = 0
    elif knee_foot_ratio < lo:
        result["knee_placement"] = 1
    else:
        result["knee_placement"] = 2
    return result


def _posture_squat_features(raw: np.ndarray) -> np.ndarray:
    rows = []
    for frame in raw:
        feature = _posture_squat_feature_frame(frame)
        if feature is not None and np.isfinite(feature).all():
            rows.append(feature)
    if not rows:
        return np.empty((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _posture_squat_feature_frame(frame: np.ndarray) -> np.ndarray | None:
    required = [0, 2, 5, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    if not np.isfinite(frame[required, :3]).all():
        return None

    points = {idx: frame[idx, :3].astype(np.float32) for idx in required}
    mid_shoulder = (points[11] + points[12]) / 2.0
    mid_hip = (points[23] + points[24]) / 2.0

    theta_neck = _safe_angle(np.array([0.0, 0.0, -1.0], dtype=np.float32), points[0] - mid_hip)
    theta_k1 = _safe_angle(points[24] - points[26], points[28] - points[26])
    theta_k2 = _safe_angle(points[23] - points[25], points[27] - points[25])
    theta_k = (theta_k1 + theta_k2) / 2.0

    theta_h1 = _safe_angle(points[26] - points[24], points[12] - points[24])
    theta_h2 = _safe_angle(points[25] - points[23], points[11] - points[23])
    theta_h = (theta_h1 + theta_h2) / 2.0

    left_tibia_length = np.linalg.norm(points[25] - points[29])
    right_tibia_length = np.linalg.norm(points[26] - points[30])
    tibia_length = (left_tibia_length + right_tibia_length) / 2.0
    if tibia_length < 1e-8:
        return None
    length_normalization_factor = (1.0 / tibia_length) ** 0.5

    z1 = (points[28][2] + points[30][2]) / 2.0 - points[32][2]
    z2 = (points[27][2] + points[29][2]) / 2.0 - points[31][2]
    z = ((z1 + z2) / 2.0) * length_normalization_factor

    left_foot_y = (points[27][1] + points[29][1] + points[31][1]) / 3.0
    right_foot_y = (points[28][1] + points[30][1] + points[32][1]) / 3.0
    ky = (((points[25][1] - left_foot_y) + (points[26][1] - right_foot_y)) / 2.0)
    ky *= length_normalization_factor

    return np.round(np.array([theta_neck, theta_k, theta_h, z, ky], dtype=np.float32), 2)


def _posture_scores_to_formai(scores: np.ndarray) -> tuple[str, list[str], float]:
    if scores.shape != (5,):
        raise ValueError("Posture scores must have shape [5]")
    correct, knee, hip, rounded_back, depth = [float(v) for v in scores]
    error_scores = {
        "knee_valgus": knee,
        "back_rounding": max(hip, rounded_back),
        "insufficient_depth": depth,
    }
    label, score = max(error_scores.items(), key=lambda item: item[1])
    if correct >= score and correct >= 0.5:
        return "good", ["Posture model predicted correct squat form"], min(max(correct, 0.5), 0.99)
    if score < 0.35:
        return "good", ["Posture model found no confident squat fault"], max(0.5, min(correct, 0.75))
    issues = [f"Posture model score {label}={score:.2f}"]
    return label, issues, min(max(score, 0.5), 0.99)


def predict_form(raw: np.ndarray, exercise: str, side: str = "right") -> FormAIPrediction:
    return ExerciseCorrectionAdapter().predict(raw, exercise, side=side)


def prediction_to_json(prediction: FormAIPrediction) -> str:
    return json.dumps(asdict(prediction), indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exercise-Correction adapter inference.")
    parser.add_argument("exercise", choices=["squat", "curl", "bicep_curl"])
    parser.add_argument("npy_path", type=Path, help="Raw MediaPipe rep tensor shaped [T,33,4].")
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument(
        "--write-metadata",
        action="store_true",
        help="Also write exercise_correction_adapter_card.json.",
    )
    args = parser.parse_args()

    adapter = ExerciseCorrectionAdapter()
    raw = np.load(args.npy_path).astype(np.float32)
    prediction = adapter.predict(raw, args.exercise, side=args.side)
    print(prediction_to_json(prediction))
    if args.write_metadata:
        adapter.write_metadata()


if __name__ == "__main__":
    main()
