//
//  FramingGuideOverlay.swift
//  FormAI
//
//  A framing guide so the user stands where Mahmoud's training videos were
//  shot (contract section 4 / Islam doc section 8). Same view = model
//  in-distribution. Auto-hides once a pose is detected.
//

import SwiftUI

struct FramingGuideOverlay: View {
    let exercise: Exercise
    let visible: Bool

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Body target box, centered, full-height portrait.
                RoundedRectangle(cornerRadius: 24)
                    .stroke(style: StrokeStyle(lineWidth: 3, dash: [12, 10]))
                    .foregroundStyle(.white.opacity(0.7))
                    .frame(width: geo.size.width * 0.55, height: geo.size.height * 0.82)
                    .position(x: geo.size.width / 2, y: geo.size.height / 2)

                VStack(spacing: 8) {
                    Image(systemName: "figure.stand")
                        .font(.system(size: 44))
                    Text("Stand 2.5–3 m back · whole body in frame")
                        .font(.subheadline.weight(.semibold))
                    Text(exercise == .squat ? "Front 3/4 view (slightly to the side)"
                                            : "Front view of the working arm")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.85))
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 24)
                .multilineTextAlignment(.center)
                .position(x: geo.size.width / 2, y: geo.size.height * 0.16)
            }
        }
        .allowsHitTesting(false)
        .opacity(visible ? 1 : 0)
        .animation(.easeInOut(duration: 0.3), value: visible)
    }
}
