# FormAI (iOS app)

Real-time exercise form coaching. Camera → MediaPipe pose (33 landmarks) →
rule-based rep counter → on each completed rep: preprocess → Core ML model →
spoken cue + on-screen form score. Exercises in scope: **bodyweight squat** and
**single-arm dumbbell curl** (contract).

This is the **app** half of the project (Islam's part). The ML models
(`SquatFormScorer`, `CurlFormScorer`) come from Syed and are **not in the repo
yet** — the whole inference path is built and wired, so dropping the models in
is the only remaining step.

## How it runs today (no model, no pod)

The project compiles and launches as-is:

- **No pose engine yet** → the camera preview and full UI work; a banner says
  the pose engine isn't linked, and no skeleton/reps are produced.
- **No model yet** → scoring automatically uses the **rule-based fallback**
  (joint-angle thresholds), so reps still get a score and a spoken cue.

Everything is behind protocols/dynamic loading so adding the pod and the models
changes no other code.

## Setup to make it fully functional

### 1. Pose engine (MediaPipe)

```bash
cd /Users/islambekdaiyn/Desktop/Projects/FormAI
pod install
open FormAI.xcworkspace      # always the workspace from now on
```

Then download **`pose_landmarker_full.task`** from the MediaPipe Pose Landmarker
model card and drag it into the `FormAI` group in Xcode (check "Copy items if
needed", target = FormAI). `MediaPipePoseProvider` loads it by that exact name.

> Same detector both sides is the whole reason this integrates — Mahmoud
> extracts training landmarks with MediaPipe Pose Landmarker too. Do **not**
> swap in Apple Vision.

### 2. The models (the only thing you'll hand me to upload)

When Syed delivers them, for **each** exercise:

1. Drag `SquatFormScorer.mlpackage` / `CurlFormScorer.mlpackage` into the
   `FormAI` group (target = FormAI). Xcode compiles them to `.mlmodelc`;
   `FormScorer` loads them at runtime by name — **no code change**.
2. Replace `FormAI/Resources/model_card_squat.json` /
   `model_card_curl.json` with the real `model_card.json` Syed ships. The
   `class_labels` array must match the model's output order exactly.

That's it — flip the **AI model** toggle on in the workout screen.

### 3. Verify the preprocessing port (the golden gate)

The preprocessing recipe (contract §7) is ported to Swift in
`FormAI/Vision/FormPreprocessor.swift`. It must match Syed's Python
byte-for-byte. Have Syed export a few preprocessed reps as
`golden_squat.json` / `golden_curl.json` (format documented in
`FormAI/SelfTest/PreprocessGoldenTest.swift`), drop them into `Resources`, and
tap **Run preprocessing self-test** on the Home screen. It must report max error
≤ 1e-3. Until a golden file is present, the button runs a synthetic shape/finite
check instead.

## Project map

| Area | Files |
|---|---|
| Pose types & config | `Models/PoseTypes.swift`, `Models/ExerciseConfig.swift` |
| Pose engine (abstracted) | `Pose/PoseProvider.swift`, `Pose/MediaPipePoseProvider.swift` |
| Contract math | `Vision/Geometry.swift`, `Vision/FormPreprocessor.swift` (§7), `Vision/RepCounter.swift` (§8) |
| Inference | `Inference/ModelCard.swift`, `Inference/FormScorer.swift`, `Inference/RuleBasedScorer.swift` |
| Voice | `Feedback/VoiceCoach.swift` |
| Orchestration | `ViewModels/WorkoutViewModel.swift` |
| UI | `Views/HomeView.swift`, `Views/WorkoutView.swift`, `Views/SkeletonOverlay.swift`, `Views/FramingGuideOverlay.swift` |
| Golden test | `SelfTest/PreprocessGoldenTest.swift` |
| Placeholder model cards | `Resources/model_card_*.json` |

## Conventions that must match Syed (documented in `FormPreprocessor.swift`)

- Image space: `x,y ∈ [0,1]`, `y` increases downward.
- `torso_lean`: angle between `(sh_mid − hip_mid)` and the **up** axis `(0,−1)`, ÷180.
- `shoulder_tilt`: angle between `(working_shoulder − other_shoulder)` and the **right** axis `(1,0)`, ÷180.
- Left-arm curl mirrors x (negate normalized x, read left indices into the right slots).

If any of these differ from Syed's reference, change them in one place
(`FormPreprocessor.swift`) and re-run the golden test.
