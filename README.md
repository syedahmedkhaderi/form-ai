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
├── models/                                 # Islam's iOS deliverables (drop into Xcode)
│   ├── SquatFormScorer.mlpackage           # Core ML model, input keypoint_window (1,32,21)
│   ├── CurlFormScorer.mlpackage            # Core ML model, input keypoint_window (1,32,11)
│   ├── SquatFormScorer_model_card.json     # feature_order, class_labels, W, F
│   ├── CurlFormScorer_model_card.json
│   └── golden_test.npy                     # 3 preprocessed squat reps for Swift validation
├── formai_pipeline.py                      # Train + export pipeline (outputs to models/)
├── exercise_correction_adapter.py          # Fallback adapter using third-party models
├── exercise_correction_adapter_card.json   # Adapter runtime metadata
├── test_exercise_correction_adapter.py     # Adapter smoke tests
├── requirements.txt                        # Pinned training deps (torch 2.4.0, coremltools 8.1)
├── requirements-posture-optional.txt       # TensorFlow for the Posture squat model
├── formai_data/                            # Synthetic training data
│   ├── squat/                              # 160 clips as [T,33,4] float32 .npy
│   ├── curl/                               # 160 clips as [T,33,4] float32 .npy
│   └── manifest.csv
├── third_party/
│   ├── Posture/                            # Pruned squat TF runtime assets
│   └── Exercise-Correction/               # Pruned curl sklearn runtime assets
├── context/                                # Team requirement documents
│   ├── 00_INTEGRATION_CONTRACTS.txt
│   ├── ISLAM_APP_REQUIREMENTS.txt
│   ├── MAHMOUD_CV_REQUIREMENTS.txt
│   └── SYED_MODEL_REQUIREMENTS.txt
└── EXERCISE_CORRECTION_INTEGRATION.md      # Additional integration notes
```

Only runtime-critical third-party files are kept: model files, licenses, READMEs,
and upstream commit markers. Demo videos, notebooks, web apps, images, and
training datasets were intentionally removed.

## Setup

```bash
python3.11 -m venv formai_env
. formai_env/bin/activate
pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir -r requirements.txt
```

For the Posture TensorFlow squat model (optional, needed by `exercise_correction_adapter.py`):

```bash
pip install -r requirements-posture-optional.txt
```

Verified stack: `torch==2.4.0+cu121`, `coremltools==8.1`, `numpy==1.26.4`, `pandas==2.2.2`, `scikit-learn==1.5.2`, `tensorflow-cpu==2.15.1`.

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

