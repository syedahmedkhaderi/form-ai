//
//  SkeletonOverlay.swift
//  FormAI
//
//  Draws the live skeleton over the camera preview. Maps MediaPipe normalized
//  coords onto the preview using the same aspect-fill transform as the preview layer.
//

import SwiftUI

struct SkeletonOverlay: View {
    let frame: PoseFrame?
    /// Pixel size of the (portrait) camera buffer.
    let imageSize: CGSize
    /// Tint for the bones (reflects current form).
    var color: Color = .green

    private let visibilityThreshold: Float = 0.3

    var body: some View {
        GeometryReader { geo in
            Canvas(rendersAsynchronously: true) { context, size in
                guard let frame, imageSize.width > 0, imageSize.height > 0 else { return }
                let transform = AspectFill.transform(image: imageSize, view: size)

                // Bones
                var path = Path()
                for (a, b) in PoseLandmarkIndex.connections {
                    let la = frame[a], lb = frame[b]
                    guard la.visibility >= visibilityThreshold, lb.visibility >= visibilityThreshold else { continue }
                    path.move(to: transform(la.point))
                    path.addLine(to: transform(lb.point))
                }
                context.stroke(path, with: .color(color), lineWidth: 4)

                // Joints
                for idx in PoseLandmarkIndex.drawnJoints {
                    let lm = frame[idx]
                    guard lm.visibility >= visibilityThreshold else { continue }
                    let p = transform(lm.point)
                    let dot = Path(ellipseIn: CGRect(x: p.x - 5, y: p.y - 5, width: 10, height: 10))
                    context.fill(dot, with: .color(.white))
                    context.stroke(dot, with: .color(color), lineWidth: 2)
                }
            }
        }
        .allowsHitTesting(false)
    }
}

/// Aspect-fill coordinate mapping from normalized image space [0,1] to a view.
enum AspectFill {
    static func transform(image: CGSize, view: CGSize) -> (CGPoint) -> CGPoint {
        let scale = max(view.width / image.width, view.height / image.height)
        let displayW = image.width * scale
        let displayH = image.height * scale
        let offsetX = (view.width - displayW) / 2
        let offsetY = (view.height - displayH) / 2
        return { normalized in
            CGPoint(x: offsetX + normalized.x * displayW,
                    y: offsetY + normalized.y * displayH)
        }
    }
}
