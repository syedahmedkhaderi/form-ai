# FormAI Exercise Form Models

FormAI provides exercise form scoring for two movements:

- **Bodyweight squat**
- **Single-arm dumbbell curl**

The project uses existing open-source model artifacts instead of training a new
model from scratch, while preserving the FormAI app contract:

```text
raw MediaPipe landmarks [T, 33, 4] -> FormAI label + logits
```

## Current Status

Both exercise paths are ready and verified through `exercise_correction_adapter.py`.

| Exercise | Runtime Source | Model Type | Status |
| --- | --- | --- | --- |
| Squat | `twixupmysleeve/Posture` | TensorFlow/Keras saved model | Ready |
| Dumbbell curl | `NgoQuocBao1010/Exercise-Correction` | scikit-learn KNN model + rules | Ready |

The adapter returns the same output shape and labels expected by the app:

```python
from exercise_correction_adapter import ExerciseCorrectionAdapter

adapter = ExerciseCorrectionAdapter()
prediction = adapter.predict(raw_landmarks, "squat")
```

## Labels

### Squat

```text
good
knee_valgus
insufficient_depth
back_rounding
```

### Dumbbell Curl

```text
good
swing
partial_rom
elbow_flare
```

## Model Architecture

### Squat

Source: [`twixupmysleeve/Posture`](third_party/Posture/UPSTREAM_COMMIT.txt)

The squat model is a TensorFlow/Keras saved model. It consumes five squat posture
features per frame:

- Neck angle
- Knee angle
- Hip angle
- Foot/depth z feature
- Knee/foot vertical feature

The model outputs five posture scores:

```text
correct, knee, hip, rounded_back, depth
```

FormAI maps those outputs as:

```text
correct -> good
knee -> knee_valgus
hip / rounded_back -> back_rounding
depth -> insufficient_depth
```

### Dumbbell Curl

Source: [`NgoQuocBao1010/Exercise-Correction`](third_party/Exercise-Correction/UPSTREAM_COMMIT.txt)

The curl path uses:

- A scikit-learn `KNeighborsClassifier` for lean-back posture detection
- Rule-based checks for partial range of motion and elbow/upper-arm flare

The sklearn model consumes 36 MediaPipe landmark features from the upper body,
arms, and hips.

## Repository Layout

```text
.
├── exercise_correction_adapter.py          # Unified runtime adapter
├── exercise_correction_adapter_card.json   # Runtime metadata
├── test_exercise_correction_adapter.py     # Adapter smoke tests
├── formai_pipeline.py                      # Original Core ML export pipeline
├── SquatFormScorer.mlpackage               # Existing Core ML package
├── CurlFormScorer.mlpackage                # Existing Core ML package
├── third_party/
│   ├── Posture/                            # Pruned squat runtime assets
│   └── Exercise-Correction/                # Pruned curl runtime assets
└── EXERCISE_CORRECTION_INTEGRATION.md      # Additional integration notes
```

Only runtime-critical third-party files are kept: model files, licenses, READMEs,
and upstream commit markers. Demo videos, notebooks, web apps, images, and
training datasets were intentionally removed.

## Setup

Use the existing virtual environment:

```bash
. formai_env/bin/activate
```

The active environment has been verified with:

- `tensorflow-cpu==2.15.1`
- `torch==2.4.0+cu121`
- `coremltools==8.1`
- `numpy==1.26.4`
- `pandas==2.2.2`
- `scikit-learn==1.5.2`

If TensorFlow needs to be installed in a fresh environment:

```bash
pip install -r requirements-posture-optional.txt
```

## Usage

### Python API

```python
import numpy as np
from exercise_correction_adapter import ExerciseCorrectionAdapter

adapter = ExerciseCorrectionAdapter()

raw = np.load("formai_data/squat/clip_0001.npy")  # shape [T, 33, 4]
prediction = adapter.predict(raw, "squat")

print(prediction.label)
print(prediction.logits)
print(prediction.source)
```

### CLI

```bash
formai_env/bin/python exercise_correction_adapter.py squat formai_data/squat/clip_0001.npy
formai_env/bin/python exercise_correction_adapter.py curl formai_data/curl/clip_0001.npy
```

## Verification

Run the test suite:

```bash
formai_env/bin/python -m unittest discover -v
```

Run dependency validation:

```bash
formai_env/bin/pip check
```

The latest full verification passed:

- All 160 squat clips routed through the Posture TensorFlow model
- All 160 curl clips routed through the Exercise-Correction curl model/rules
- No invalid logits or label mismatches
- Existing FormAI preprocessing still returns finite tensors
- Core ML conversion smoke test succeeded

## Notes

- The adapter is Python-connectable today.
- The existing Core ML packages remain in the repository.
- For direct on-device iOS use of the third-party models, add a backend call,
  Swift port, or conversion layer while preserving the same label/logit contract.

