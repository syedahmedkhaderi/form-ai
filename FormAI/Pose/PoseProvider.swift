//
//  PoseProvider.swift
//  FormAI
//
//  Protocol and factory for the pose engine. The app uses MediaPipe's 33-landmark
//  schema, so all feature extraction assumes those indices. NoopPoseProvider lets
//  the project build and run before MediaPipeTasksVision is installed.
//

import Foundation
import CoreVideo

/// Receives pose results as frames stream in (live-stream mode is async).
protocol PoseProviderDelegate: AnyObject {
    /// Called (on an arbitrary queue) when a frame has been processed.
    /// `frame` is nil if no pose was detected in that image.
    func poseProvider(_ provider: PoseProvider,
                      didDetect frame: PoseFrame?,
                      imageSize: CGSize,
                      timestampMs: Int)
}

/// A live-stream pose engine.
protocol PoseProvider: AnyObject {
    var delegate: PoseProviderDelegate? { get set }

    /// Whether the engine is actually available/loaded. When false the app
    /// shows a banner and only the camera preview works.
    var isAvailable: Bool { get }

    /// Human-readable status (shown in the UI when unavailable).
    var statusMessage: String { get }

    /// Feed one camera frame. `timestampMs` must be monotonically increasing
    /// (MediaPipe live-stream requirement).
    func process(pixelBuffer: CVPixelBuffer, timestampMs: Int)
}

/// Stub used when MediaPipeTasksVision is not linked yet. Lets the project
/// compile and the camera UI run; produces no landmarks.
final class NoopPoseProvider: PoseProvider {
    weak var delegate: PoseProviderDelegate?
    var isAvailable: Bool { false }
    var statusMessage: String {
        "Pose engine not installed. Run `pod install`, add MediaPipeTasksVision "
        + "and pose_landmarker_full.task, then rebuild."
    }

    func process(pixelBuffer: CVPixelBuffer, timestampMs: Int) {
        // Intentionally does nothing.
    }
}

/// Factory that returns the best available provider.
enum PoseEngine {
    static func makeProvider() -> PoseProvider {
        #if canImport(MediaPipeTasksVision)
        return MediaPipePoseProvider()
        #else
        return NoopPoseProvider()
        #endif
    }
}
