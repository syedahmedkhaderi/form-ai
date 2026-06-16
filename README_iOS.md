# FormAI (iOS app)

Real-time exercise form coaching. Camera → MediaPipe pose (33 landmarks) →
rule-based rep counter → on each completed rep: preprocess → Core ML model →
spoken cue + on-screen form score. Exercises in scope: **bodyweight squat** and
**single-arm dumbbell curl** (contract).

This is the **app** half of the project (Islam's part). The ML models
(`SquatFormScorer`, `CurlFormScorer`) are bundled under `FormAI/Resources`, and
the inference path loads them dynamically at runtime.

## How it runs today

The project is set up with CocoaPods and bundled model assets:

- **Pose engine** → `MediaPipeTasksVision` is declared in the Podfile and
  installed with `pod install`. `pose_landmarker_full.task` is bundled.
- **Models** → `SquatFormScorer.mlpackage` and `CurlFormScorer.mlpackage` are
  present under `FormAI/Resources`, with matching model cards.
- **Fallback** → if a model fails to load, scoring still falls back to
  joint-angle rules.

Open `FormAI.xcworkspace`, not `FormAI.xcodeproj`, after installing pods.

## Setup to run the app

### 1. Install iOS tooling

Install full Xcode from the Mac App Store or Apple Developer Downloads. Command
Line Tools alone are not enough for `xcodebuild` or simulator/device builds.

After installing Xcode:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept
```

### 2. Install pods

```bash
cd /Users/syed/Downloads/form-ai-1
pod install
open FormAI.xcworkspace
```

### 3. Verify the preprocessing port

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
