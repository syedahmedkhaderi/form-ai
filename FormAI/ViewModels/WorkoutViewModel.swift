//
//  WorkoutViewModel.swift
//  FormAI
//
//  Orchestrates the whole loop (Islam doc section 5):
//  camera -> MediaPipe pose -> rep counter -> on rep complete: preprocess ->
//  Core ML model (or rule-based fallback) -> spoken cue + on-screen score.
//

import Foundation
import SwiftUI
import Combine
import AVFoundation

@MainActor
final class WorkoutViewModel: ObservableObject {
    // Selection
    @Published private(set) var exercise: Exercise
    @Published private(set) var arm: Arm

    // Live state
    @Published var repCount = 0
    @Published var formScore: Int = 0
    @Published var lastLabel: String = ""
    @Published var lastCue: String = "Get in frame to begin"
    @Published var phase: RepPhase = .idle
    @Published var currentAngle: Float = .nan
    @Published var liveState: LiveCoachingState = .outOfFrame
    @Published var liveMessage: String = "Get in frame to begin"
    @Published var liveSeverity: LiveCoachingSeverity = .neutral
    @Published private(set) var currentFrame: PoseFrame?
    @Published private(set) var imageSize: CGSize = .zero

    // Status
    @Published private(set) var usingFrontCamera = false
    @Published var useModel = true
    @Published var voiceEnabled = true { didSet { voiceCoach.isEnabled = voiceEnabled } }
    @Published private(set) var isRunning = false
    @Published private(set) var cameraDenied = false
    @Published private(set) var poseStatus: String = ""
    @Published private(set) var poseAvailable = false
    @Published private(set) var modelStatus: String = ""
    @Published private(set) var modelLoaded = false
    @Published var goldenTestMessage: String = ""

    private let cameraManager = CameraManager()
    private let poseProvider: PoseProvider
    private let voiceCoach = VoiceCoach()
    private var repCounter: RepCounter
    private var scorer: FormScorer
    private let historyStore = WorkoutHistoryStore.shared
    private var recentFrames: [PoseFrame] = []
    private var stableFrameCount = 0
    private var liveFrameCounter = 0
    private var overlayFrameCounter = 0
    private var lastLiveCandidateLabel = ""
    private var liveCandidateStableCount = 0
    private var lastLiveWarningLabel = ""
    private var lastLiveVoiceAt: Date = .distantPast
    private var sessionScores: [Int] = []
    private var hasPersistedWorkout = false

    private let liveWindowSize = 18
    private let liveStride = 4
    private let overlayStride = 2
    private let stableFramesRequired = 6
    private let liveSpeechCooldown: TimeInterval = 2.5

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
        lastLabel = ""
        lastCue = "Get in frame to begin"
        phase = .idle
        currentFrame = nil
        liveState = .outOfFrame
        liveMessage = "Get in frame to begin"
        liveSeverity = .neutral
        recentFrames.removeAll(keepingCapacity: true)
        stableFrameCount = 0
        liveFrameCounter = 0
        overlayFrameCounter = 0
        lastLiveCandidateLabel = ""
        liveCandidateStableCount = 0
        lastLiveWarningLabel = ""
        lastLiveVoiceAt = .distantPast
        sessionScores.removeAll(keepingCapacity: true)
        hasPersistedWorkout = false
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
        persistWorkoutSummaryIfNeeded()
        cameraManager.stop()
        voiceCoach.stop()
        isRunning = false
        currentFrame = nil
    }

    /// Flip between front and back camera.
    func flipCamera() {
        cameraManager.switchCamera()
        usingFrontCamera.toggle()
    }

    // MARK: - Golden test

    func runGoldenTest() {
        let result = PreprocessGoldenTest.run(for: exercise)
        goldenTestMessage = result.message
    }

    // MARK: - Scoring a completed rep

    private func handleCompletedRep(_ frames: [PoseFrame]) {
        var result: ScoreResult?

        if useModel, scorer.isLoaded,
           let window = FormPreprocessor.preprocess(frames: frames, exercise: exercise, arm: arm) {
            result = scorer.score(window: window)
        }
        if result == nil {
            result = RuleBasedScorer.score(frames: frames, exercise: exercise, arm: arm)
        }

        guard let r = result else { return }
        formScore = r.score
        lastLabel = r.label
        sessionScores.append(r.score)
        let cue = Cue.text(for: r.label, exercise: exercise)
        lastCue = cue
        liveState = .repComplete
        liveMessage = "Last rep scored \(r.score). \(cue)"
        liveSeverity = Cue.isGood(r.label) ? .success : .caution
        lastLiveCandidateLabel = ""
        liveCandidateStableCount = 0
        voiceCoach.speak(cue, interrupt: true)
    }

    private func persistWorkoutSummaryIfNeeded() {
        guard !hasPersistedWorkout, repCount > 0 else { return }
        let averageScore = sessionScores.isEmpty ? formScore : Int((Float(sessionScores.reduce(0, +)) / Float(sessionScores.count)).rounded())
        historyStore.record(
            WorkoutSummary(
                exercise: exercise,
                reps: repCount,
                averageScore: averageScore,
                finalScore: formScore
            )
        )
        hasPersistedWorkout = true
    }

    private func setLiveState(_ state: LiveCoachingState, message: String, severity: LiveCoachingSeverity) {
        guard liveState != state || liveMessage != message || liveSeverity != severity else { return }
        liveState = state
        liveMessage = message
        liveSeverity = severity
    }

    private func updateRepStatus(from event: RepEvent) {
        if repCount != event.count {
            repCount = event.count
        }
        if phase != event.phase {
            phase = event.phase
        }
        if currentAngle.isNaN != event.currentAngle.isNaN || (!currentAngle.isNaN && abs(currentAngle - event.currentAngle) >= 0.5) {
            currentAngle = event.currentAngle
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
        let bufferSize = cameraManager.bufferSize
        DispatchQueue.main.async { [weak self] in
            self?.ingest(frame: frame, bufferSize: bufferSize)
        }
    }

    private func ingest(frame: PoseFrame?, bufferSize: CGSize) {
        if imageSize != bufferSize {
            imageSize = bufferSize
        }
        guard let frame else {
            stableFrameCount = 0
            overlayFrameCounter = 0
            recentFrames.removeAll(keepingCapacity: true)
            currentFrame = nil
            setLiveState(.outOfFrame, message: "Get in frame to begin.", severity: .neutral)
            return
        }

        stableFrameCount += 1
        liveFrameCounter += 1
        overlayFrameCounter += 1

        if currentFrame == nil || overlayFrameCounter % overlayStride == 0 {
            currentFrame = frame
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
