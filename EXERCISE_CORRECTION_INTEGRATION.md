# Exercise-Correction Integration

This project now vendors `NgoQuocBao1010/Exercise-Correction` under
`third_party/Exercise-Correction`, vendors `twixupmysleeve/Posture` under
`third_party/Posture`, and exposes both through `exercise_correction_adapter.py`.

## Why an Adapter Exists

The upstream repo is not a drop-in replacement for the existing Core ML training
pipeline:

- upstream inference is frame-based MediaPipe Pose plus sklearn pickle models;
- Posture squat inference is a TensorFlow saved model that predicts five
  squat feedback flags;
- FormAI consumes rep clips shaped `[T, 33, 4]`;
- the iOS contract expects stable labels/logits for `squat` and `curl`;
- Exercise-Correction's old squat stage pickle is intentionally not part of the
  runtime because squat is now handled by Posture.

The adapter keeps the app-facing surface stable and avoids training new model
weights. Squat uses Posture's saved TensorFlow model. Curl uses the
Exercise-Correction bicep curl sklearn model plus deterministic angle checks.

## Python API

```python
import numpy as np
from exercise_correction_adapter import ExerciseCorrectionAdapter

adapter = ExerciseCorrectionAdapter()
raw = np.load("formai_data/squat/clip_0001.npy")  # [T, 33, 4]
prediction = adapter.predict(raw, "squat")

print(prediction.label)
print(prediction.logits)
print(prediction.issues)
```

`prediction` is a `FormAIPrediction` dataclass with:

- `exercise`
- `label`
- `class_labels`
- `logits`
- `confidence`
- `issues`
- `source`
- `model_available`
- `model_load_error`

## Label Mapping

Squat labels remain compatible with FormAI and are sourced from Posture:

```text
good, knee_valgus, insufficient_depth, back_rounding
```

Posture squat support:

- model input: five per-frame squat features from Posture's `SquatPosture.py`;
- model output: `correct`, `knee`, `hip`, `rounded_back`, `depth`;
- FormAI mapping: `correct -> good`, `knee -> knee_valgus`,
  `hip/rounded_back -> back_rounding`, `depth -> insufficient_depth`.

If TensorFlow is unavailable, squat still returns compatible predictions using a
threshold fallback and records the Posture load error in metadata. The committed
environment installs `tensorflow-cpu==2.15.1`, so the normal path is the
Posture model.

Curl labels remain compatible with FormAI:

```text
good, swing, partial_rom, elbow_flare
```

Upstream curl support:

- sklearn model: lean-back posture, mapped to `swing`;
- deterministic rules: weak peak contraction mapped to `partial_rom`;
- deterministic rules: loose upper arm mapped to `elbow_flare`.

## Posture TensorFlow Runtime

Install the Posture runtime dependency with:

```bash
pip install -r requirements-posture-optional.txt
```

The adapter itself does not require OpenCV or MediaPipe for Posture inference;
it consumes the existing raw `[T, 33, 4]` MediaPipe tensor and recreates
Posture's five squat features in NumPy.

## Verification

Run:

```bash
formai_env/bin/python -m unittest test_exercise_correction_adapter.py
```

Optional metadata:

```python
from exercise_correction_adapter import ExerciseCorrectionAdapter
ExerciseCorrectionAdapter().write_metadata()
```
