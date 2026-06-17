//
//  MediaPipePoseProvider.swift
//  FormAI
//
//  MediaPipe Pose Landmarker in LIVE_STREAM mode. Compiled only when the
//  MediaPipeTasksVision pod is linked. Everything else in the app talks to
//  the PoseProvider protocol, so the pod can be swapped without downstream changes.
//
//  Setup (see README / Podfile):
//    - pod 'MediaPipeTasksVision'
//    - bundle pose_landmarker_full.task into the app target
//

#if canImport(MediaPipeTasksVision)
import Foundation
import CoreVideo
import MediaPipeTasksVision

final class MediaPipePoseProvider: NSObject, PoseProvider {
    weak var delegate: PoseProviderDelegate?

    private var landmarker: PoseLandmarker?
    private var loadError: String?

    /// Name of the task file bundled into the app ("full" variant for better accuracy).
    private static let taskFileName = "pose_landmarker_full"

    var isAvailable: Bool { landmarker != nil }

    var statusMessage: String {
        if isAvailable { return "MediaPipe Pose Landmarker ready." }
        return loadError ?? "MediaPipe Pose Landmarker failed to load."
    }

    override init() {
        super.init()
        configure()
    }

    private func configure() {
        guard let modelPath = Bundle.main.path(forResource: Self.taskFileName, ofType: "task") else {
            loadError = "Missing \(Self.taskFileName).task in the app bundle. "
                + "Download it from the MediaPipe Pose Landmarker model card and add it to the FormAI target."
            return
        }

        let options = PoseLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = .liveStream
        options.numPoses = 1
        options.minPoseDetectionConfidence = 0.5
        options.minPosePresenceConfidence = 0.5
        options.minTrackingConfidence = 0.5
        options.poseLandmarkerLiveStreamDelegate = self

        do {
            landmarker = try PoseLandmarker(options: options)
        } catch {
            loadError = "Could not create PoseLandmarker: \(error.localizedDescription)"
        }
    }

    func process(pixelBuffer: CVPixelBuffer, timestampMs: Int) {
        guard let landmarker else { return }
        do {
            // The camera connection is rotated to portrait-upright (see
            // CameraManager), so the buffer is already display-oriented.
            let image = try MPImage(pixelBuffer: pixelBuffer, orientation: .up)
            try landmarker.detectAsync(image: image, timestampInMilliseconds: timestampMs)
        } catch {
            // Dropped frames / out-of-order timestamps are non-fatal in live stream.
        }
    }
}

extension MediaPipePoseProvider: PoseLandmarkerLiveStreamDelegate {
    func poseLandmarker(_ poseLandmarker: PoseLandmarker,
                        didFinishDetection result: PoseLandmarkerResult?,
                        timestampInMilliseconds: Int,
                        error: Error?) {
        guard let result, let first = result.landmarks.first, first.count == PoseLandmarkIndex.count else {
            delegate?.poseProvider(self, didDetect: nil, imageSize: .zero, timestampMs: timestampInMilliseconds)
            return
        }

        let landmarks: [Landmark] = first.map { lm in
            Landmark(x: lm.x,
                     y: lm.y,
                     z: lm.z,
                     visibility: lm.visibility?.floatValue ?? 0)
        }

        // Normalized coordinates are image-relative, so image size only matters
        // for aspect-fill overlay mapping; the camera reports actual dimensions
        // separately. Report a unit size here and let CameraManager carry the
        // true buffer dimensions to the overlay.
        delegate?.poseProvider(self,
                               didDetect: PoseFrame(landmarks: landmarks),
                               imageSize: CGSize(width: 1, height: 1),
                               timestampMs: timestampInMilliseconds)
    }
}
#endif
