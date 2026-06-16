#!/usr/bin/env python3
"""Smoke tests for the Exercise-Correction adapter."""

import unittest
from pathlib import Path

import numpy as np

from exercise_correction_adapter import (
    CURL_LABELS,
    SQUAT_LABELS,
    ExerciseCorrectionAdapter,
    _posture_scores_to_formai,
    _posture_squat_features,
)


class ExerciseCorrectionAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = ExerciseCorrectionAdapter()

    def test_squat_prediction_matches_formai_contract(self):
        raw = np.load(Path("formai_data") / "squat" / "clip_0001.npy")
        prediction = self.adapter.predict(raw, "squat")

        self.assertEqual(prediction.exercise, "squat")
        self.assertEqual(prediction.class_labels, SQUAT_LABELS)
        self.assertIn(prediction.label, SQUAT_LABELS)
        self.assertEqual(len(prediction.logits), len(SQUAT_LABELS))
        self.assertTrue(np.isfinite(prediction.logits).all())
        self.assertTrue(
            prediction.source.startswith("Posture")
            or prediction.source.startswith("Exercise-Correction")
        )

    def test_curl_prediction_matches_formai_contract(self):
        raw = np.load(Path("formai_data") / "curl" / "clip_0001.npy")
        prediction = self.adapter.predict(raw, "curl")

        self.assertEqual(prediction.exercise, "curl")
        self.assertEqual(prediction.class_labels, CURL_LABELS)
        self.assertIn(prediction.label, CURL_LABELS)
        self.assertEqual(len(prediction.logits), len(CURL_LABELS))
        self.assertTrue(np.isfinite(prediction.logits).all())
        if prediction.model_available:
            self.assertIn("bicep sklearn posture model", prediction.source)

    def test_invalid_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.predict(np.zeros((32, 12), dtype=np.float32), "squat")

    def test_zero_width_squat_feet_are_unknown_not_error(self):
        raw = np.load(Path("formai_data") / "squat" / "clip_0001.npy")
        raw[:, 31, :2] = raw[:, 32, :2]

        prediction = self.adapter.predict(raw, "squat")

        self.assertNotIn("knee placement outside upstream thresholds", prediction.issues)

    def test_posture_squat_features_are_finite(self):
        raw = np.load(Path("formai_data") / "squat" / "clip_0001.npy")
        features = _posture_squat_features(raw)

        self.assertEqual(features.shape[1], 5)
        self.assertGreater(features.shape[0], 0)
        self.assertTrue(np.isfinite(features).all())

    def test_posture_score_mapping_matches_formai_labels(self):
        cases = [
            (np.array([0.8, 0.1, 0.1, 0.1, 0.1], dtype=np.float32), "good"),
            (np.array([0.1, 0.9, 0.1, 0.1, 0.1], dtype=np.float32), "knee_valgus"),
            (np.array([0.1, 0.1, 0.9, 0.1, 0.1], dtype=np.float32), "back_rounding"),
            (np.array([0.1, 0.1, 0.1, 0.9, 0.1], dtype=np.float32), "back_rounding"),
            (np.array([0.1, 0.1, 0.1, 0.1, 0.9], dtype=np.float32), "insufficient_depth"),
        ]
        for scores, expected in cases:
            label, _, _ = _posture_scores_to_formai(scores)
            self.assertEqual(label, expected)


if __name__ == "__main__":
    unittest.main()
