//
//  CameraPreview.swift
//  FormAI
//
//  SwiftUI wrapper around AVCaptureVideoPreviewLayer. Aspect-fill so the
//  preview matches how the overlay maps landmarks.
//

import SwiftUI
import AVFoundation

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        disableMirroring(view.videoPreviewLayer)
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        // The preview connection is recreated when the camera flips; keep it
        // non-mirrored so it matches the (non-mirrored) buffer and skeleton.
        disableMirroring(uiView.videoPreviewLayer)
    }

    private func disableMirroring(_ layer: AVCaptureVideoPreviewLayer) {
        guard let connection = layer.connection, connection.isVideoMirroringSupported else { return }
        connection.automaticallyAdjustsVideoMirroring = false
        connection.isVideoMirrored = false
    }

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }
}
