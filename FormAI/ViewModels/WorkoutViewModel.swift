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
        let cue = Cue.text(for: r.label, exercise: exercise)
        lastCue = cue
        voiceCoach.speak(cue, interrupt: true)
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
        imageSize = bufferSize
        currentFrame = frame
        guard let frame else { return }

        let event = repCounter.consume(frame)
        repCount = event.count
        phase = event.phase
        currentAngle = event.currentAngle
        if let rep = event.completedRep {
            handleCompletedRep(rep)
        }
    }
}
