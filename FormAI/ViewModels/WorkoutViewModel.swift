//
//  WorkoutViewModel.swift
//  FormAI
//
//  Orchestrates the whole loop (Islam doc section 5):
//  camera -> MediaPipe pose -> rep counter -> on rep complete: preprocess ->
//  Core ML model (or rule-based fallback) -> spoken cue + on-screen score.
//

import Foundation
import Combine
import AVFoundation

private final class FrameDispatchGate {
    private let lock = NSLock()
    private var isScheduled = false
    private var lastAcceptedTimestampMs = Int.min

    func trySchedule(timestampMs: Int, minIntervalMs: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if isScheduled { return false }
        if lastAcceptedTimestampMs != Int.min, timestampMs - lastAcceptedTimestampMs < minIntervalMs {
            return false
        }
        isScheduled = true
        lastAcceptedTimestampMs = timestampMs
        return true
    }

    func finish() {
        lock.lock()
        isScheduled = false
        lock.unlock()
    }
}

@MainActor
final class WorkoutViewModel: ObservableObject {
    // Selection
    @Published private(set) var exercise: Exercise
    @Published private(set) var arm: Arm

    // Live state
    @Published var repCount = 0
    @Published var formScore: Int = 0
    @Published var liveState: LiveCoachingState = .outOfFrame
    @Published var liveMessage: String = "Get in frame to begin"
    @Published var liveSeverity: LiveCoachingSeverity = .neutral
    @Published private(set) var hasPoseInFrame = false

    // Status
    @Published private(set) var usingFrontCamera = false
    @Published var voiceEnabled = true { didSet { voiceCoach.isEnabled = voiceEnabled } }
    @Published private(set) var isRunning = false
    @Published private(set) var cameraDenied = false
    @Published private(set) var poseStatus: String = ""
    @Published private(set) var poseAvailable = false
    @Published private(set) var modelStatus: String = ""
    @Published private(set) var modelLoaded = false

    private let cameraManager = CameraManager()
    private let poseProvider: PoseProvider
    private let voiceCoach = VoiceCoach()
    private var repCounter: RepCounter
    private var scorer: FormScorer
    nonisolated(unsafe) private let frameDispatchGate = FrameDispatchGate()
    private var recentFrames: [PoseFrame] = []
    private var stableFrameCount = 0
    private var liveFrameCounter = 0
    private var lastLiveCandidateLabel = ""
    private var liveCandidateStableCount = 0
    private var lastLiveWarningLabel = ""
    private var lastLiveVoiceAt: Date = .distantPast

    private let liveWindowSize = 18
    private let liveStride = 4
    private let stableFramesRequired = 6
    private let liveSpeechCooldown: TimeInterval = 2.5
    private let ingestIntervalMs = 50
    private let landmarkVisibilityThreshold: Float = 0.45

    var session: AVCaptureSession { cameraManager.session }

    init(exercise: Exercise = .squat, arm: Arm = .right) {
        self.exercise = exercise
        self.arm = arm
        self.poseProvider = PoseEngine.makeProvider()
        self.repCounter = RepCounter(exercise: exercise, arm: arm)
        self.scorer = FormScorer(exercise: exercise)

        poseProvider.delegate = self
        poseAvailable = poseProvider.isAvailable
        poseStatus = poseProvider.statusMessage
        modelLoaded = scorer.isLoaded
        modelStatus = scorer.loadMessage

        let provider = poseProvider
        cameraManager.onFrame = { buffer, ts in
            provider.process(pixelBuffer: buffer, timestampMs: ts)
        }
        recentFrames.reserveCapacity(liveWindowSize)
    }

    // MARK: - Configuration

    func configure(exercise: Exercise, arm: Arm) {
        self.exercise = exercise
        self.arm = arm
        repCounter = RepCounter(exercise: exercise, arm: arm)
        scorer = FormScorer(exercise: exercise)
        modelLoaded = scorer.isLoaded
        modelStatus = scorer.loadMessage
        resetMetrics()
    }

    private func resetMetrics() {
        repCounter.reset()
        repCount = 0
        formScore = 0
        hasPoseInFrame = false
        liveState = .outOfFrame
        liveMessage = "Get in frame to begin"
        liveSeverity = .neutral
        recentFrames.removeAll(keepingCapacity: true)
        stableFrameCount = 0
        liveFrameCounter = 0
        lastLiveCandidateLabel = ""
        liveCandidateStableCount = 0
        lastLiveWarningLabel = ""
        lastLiveVoiceAt = .distantPast
    }

    // MARK: - Lifecycle

    func start() async {
        let granted = await CameraManager.requestAccess()
        cameraDenied = !granted
        guard granted else { return }
        resetMetrics()
        cameraManager.start()
        isRunning = true
    }

    func stop() {
        cameraManager.stop()
        voiceCoach.stop()
        isRunning = false
        hasPoseInFrame = false
    }

    /// Flip between front and back camera.
    func flipCamera() {
        cameraManager.switchCamera()
        usingFrontCamera.toggle()
    }

    // MARK: - Scoring a completed rep

    private func handleCompletedRep(_ frames: [PoseFrame]) {
        var result: ScoreResult?

        if scorer.isLoaded,
           let window = FormPreprocessor.preprocess(frames: frames, exercise: exercise, arm: arm) {
            result = scorer.score(window: window)
        }
        if result == nil {
            result = RuleBasedScorer.score(frames: frames, exercise: exercise, arm: arm)
        }

        guard let r = result else { return }
        formScore = r.score
        let cue = Cue.text(for: r.label, exercise: exercise)
        setLiveState(.repComplete, message: "Last rep scored \(r.score). \(cue)", severity: Cue.isGood(r.label) ? .success : .caution)
        lastLiveCandidateLabel = ""
        liveCandidateStableCount = 0
        voiceCoach.speak(cue, interrupt: true)
    }

    private func setLiveState(_ state: LiveCoachingState, message: String, severity: LiveCoachingSeverity) {
        guard liveState != state || liveMessage != message || liveSeverity != severity else { return }
        liveState = state
        liveMessage = message
        liveSeverity = severity
    }

    private func isFrameUsable(_ frame: PoseFrame) -> Bool {
        let indices: [Int]
        switch exercise {
        case .squat:
            indices = [
                PoseLandmarkIndex.leftShoulder, PoseLandmarkIndex.rightShoulder,
                PoseLandmarkIndex.leftHip, PoseLandmarkIndex.rightHip,
                PoseLandmarkIndex.leftKnee, PoseLandmarkIndex.rightKnee,
                PoseLandmarkIndex.leftAnkle, PoseLandmarkIndex.rightAnkle
            ]
        case .curl:
            let elbow = arm == .right ? PoseLandmarkIndex.rightElbow : PoseLandmarkIndex.leftElbow
            let wrist = arm == .right ? PoseLandmarkIndex.rightWrist : PoseLandmarkIndex.leftWrist
            let shoulder = arm == .right ? PoseLandmarkIndex.rightShoulder : PoseLandmarkIndex.leftShoulder
            let hip = arm == .right ? PoseLandmarkIndex.rightHip : PoseLandmarkIndex.leftHip
            indices = [
                shoulder, elbow, wrist, hip,
                PoseLandmarkIndex.leftShoulder, PoseLandmarkIndex.rightShoulder
            ]
        }

        return indices.allSatisfy { frame[$0].visibility >= landmarkVisibilityThreshold }
    }

    private func updateRepStatus(from event: RepEvent) {
        if repCount != event.count {
            repCount = event.count
        }
    }

    private func updateLiveCoaching(for event: RepEvent) {
        guard stableFrameCount >= stableFramesRequired else {
            setLiveState(.stabilizing, message: "Hold still for a second so tracking can settle.", severity: .neutral)
            return
        }

        switch event.progress {
        case .searching:
            setLiveState(.ready, message: "Stand tall at the start position to begin.", severity: .neutral)

        case .ready:
            setLiveState(.ready, message: "Pose is locked. Start the rep when ready.", severity: .neutral)

        case .descending, .bottom, .rising:
            setLiveState(.repActive, message: "Rep started. Move with control.", severity: .neutral)
            guard liveFrameCounter % liveStride == 0 else { return }
            guard let evaluation = RuleBasedScorer.liveEvaluation(
                frames: recentFrames,
                exercise: exercise,
                arm: arm,
                progress: event.progress
            ) else { return }

            if evaluation.label == lastLiveCandidateLabel {
                liveCandidateStableCount += 1
            } else {
                lastLiveCandidateLabel = evaluation.label
                liveCandidateStableCount = 1
            }

            guard liveCandidateStableCount >= 2 else { return }

            if evaluation.label == "good" {
                setLiveState(.goodMovement, message: evaluation.message, severity: evaluation.severity)
                return
            }

            setLiveState(.warning(label: evaluation.label), message: evaluation.message, severity: evaluation.severity)

            let now = Date()
            let shouldSpeak = evaluation.shouldSpeak
                && evaluation.label != lastLiveWarningLabel
                && now.timeIntervalSince(lastLiveVoiceAt) >= liveSpeechCooldown
            if shouldSpeak {
                voiceCoach.speak(evaluation.message)
                lastLiveWarningLabel = evaluation.label
                lastLiveVoiceAt = now
            }

        case .completed:
            break
        }
    }
}

// MARK: - PoseProviderDelegate

extension WorkoutViewModel: PoseProviderDelegate {
    nonisolated func poseProvider(_ provider: PoseProvider,
                                  didDetect frame: PoseFrame?,
                                  imageSize: CGSize,
                                  timestampMs: Int) {
        guard frameDispatchGate.trySchedule(timestampMs: timestampMs, minIntervalMs: ingestIntervalMs) else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            defer { self.frameDispatchGate.finish() }
            self.ingest(frame: frame)
        }
    }

    private func ingest(frame: PoseFrame?) {
        guard let frame else {
            stableFrameCount = 0
            recentFrames.removeAll(keepingCapacity: true)
            if hasPoseInFrame {
                hasPoseInFrame = false
            }
            setLiveState(.outOfFrame, message: "Get in frame to begin.", severity: .neutral)
            return
        }

        guard isFrameUsable(frame) else {
            stableFrameCount = 0
            recentFrames.removeAll(keepingCapacity: true)
            if hasPoseInFrame {
                hasPoseInFrame = false
            }
            setLiveState(.stabilizing, message: "Hold still and keep the key joints visible.", severity: .neutral)
            return
        }

        stableFrameCount += 1
        liveFrameCounter += 1
        if !hasPoseInFrame {
            hasPoseInFrame = true
        }

        if recentFrames.count == liveWindowSize {
            recentFrames.removeFirst()
        }
        recentFrames.append(frame)

        let event = repCounter.consume(frame)
        updateRepStatus(from: event)
        updateLiveCoaching(for: event)
        if let rep = event.completedRep {
            handleCompletedRep(rep)
        }
    }
}
