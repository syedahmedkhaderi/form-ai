//
//  CameraManager.swift
//  FormAI
//
//  AVFoundation capture (Islam doc section 5a). Streams video frames from the
//  back camera, rotated to portrait-upright so MediaPipe receives a
//  display-oriented image and normalized landmarks map straight onto the
//  portrait preview.
//

import Foundation
import AVFoundation

final class CameraManager: NSObject {
    let session = AVCaptureSession()

    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "formai.camera.session")
    private let videoQueue = DispatchQueue(label: "formai.camera.video")

    /// Called on the video queue for each frame. (pixelBuffer, timestampMs).
    nonisolated(unsafe) var onFrame: ((CVPixelBuffer, Int) -> Void)?

    /// Pixel dimensions of the (rotated) frames, for overlay aspect-fill mapping.
    nonisolated(unsafe) private(set) var bufferSize: CGSize = .zero

    /// Current camera position (back by default; phone on a stand).
    nonisolated(unsafe) private(set) var position: AVCaptureDevice.Position = .back

    private var configured = false
    private var videoInput: AVCaptureDeviceInput?

    // MARK: - Permission

    static func requestAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    // MARK: - Lifecycle

    func start() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if !self.configured { self.configure() }
            if !self.session.isRunning { self.session.startRunning() }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    private func configure() {
        configured = true
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }

        applyInput(for: position)
        session.commitConfiguration()
    }

    /// Flip between front and back camera.
    func switchCamera() {
        sessionQueue.async { [weak self] in
            guard let self, self.configured else { return }
            let next: AVCaptureDevice.Position = self.position == .back ? .front : .back
            self.session.beginConfiguration()
            self.applyInput(for: next)
            self.session.commitConfiguration()
        }
    }

    /// (Re)attach the video input for the given position and re-apply orientation.
    /// Must be called inside a begin/commit configuration block.
    private func applyInput(for newPosition: AVCaptureDevice.Position) {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: newPosition),
              let input = try? AVCaptureDeviceInput(device: device) else {
            return
        }

        if let existing = videoInput {
            session.removeInput(existing)
        }
        guard session.canAddInput(input) else {
            // Roll back to the previous input if the new one can't be added.
            if let existing = videoInput { session.addInput(existing) }
            return
        }
        session.addInput(input)
        videoInput = input
        position = newPosition

        // Rotate the output to portrait-upright (90 degrees). Keep the image
        // NON-mirrored on the front camera so left/right (and valgus) stay
        // physically correct for the model and the overlay aligns.
        if let connection = videoOutput.connection(with: .video) {
            if #available(iOS 17.0, *) {
                let angle: CGFloat = 90
                if connection.isVideoRotationAngleSupported(angle) {
                    connection.videoRotationAngle = angle
                }
            } else {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = false
            }
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput,
                                   didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        bufferSize = CGSize(width: w, height: h)

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let ms = Int(CMTimeGetSeconds(pts) * 1000)
        onFrame?(pixelBuffer, ms)
    }
}
