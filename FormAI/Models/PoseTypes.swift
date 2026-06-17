//
//  PoseTypes.swift
//  FormAI
//
//  Canonical pose data types shared across the pipeline.
//  MediaPipe Pose: 33 landmarks, indices 0..32, each (x, y, z, visibility),
//  with x/y normalized to [0,1] relative to image width/height.
//
//  Using our own types instead of MediaPipe's directly keeps the rest of the
//  app decoupled from the pose engine.
//

import Foundation
import CoreGraphics

/// A single landmark. `x`/`y` are normalized to [0,1]. `z` is depth (noisy,
/// dropped in features but kept here per the contract). `visibility` in [0,1].
struct Landmark: Equatable {
    var x: Float
    var y: Float
    var z: Float
    var visibility: Float

    /// 2D point in normalized image space.
    var point: CGPoint { CGPoint(x: CGFloat(x), y: CGFloat(y)) }

    static let zero = Landmark(x: 0, y: 0, z: 0, visibility: 0)
}

/// One frame = 33 landmarks. Index `i` is MediaPipe landmark `i`.
struct PoseFrame: Equatable {
    /// Always exactly 33 entries.
    var landmarks: [Landmark]

    init(landmarks: [Landmark]) {
        precondition(landmarks.count == PoseLandmarkIndex.count,
                     "PoseFrame requires exactly \(PoseLandmarkIndex.count) landmarks")
        self.landmarks = landmarks
    }

    subscript(_ index: Int) -> Landmark { landmarks[index] }
}

/// MediaPipe Pose landmark indices used by FormAI.
/// `count` is the full landmark count (33), not just the used subset.
enum PoseLandmarkIndex {
    static let count = 33

    static let nose = 0
    static let leftShoulder = 11
    static let rightShoulder = 12
    static let leftElbow = 13
    static let rightElbow = 14
    static let leftWrist = 15
    static let rightWrist = 16
    static let leftHip = 23
    static let rightHip = 24
    static let leftKnee = 25
    static let rightKnee = 26
    static let leftAnkle = 27
    static let rightAnkle = 28

    /// Connections drawn by the skeleton overlay (pairs of landmark indices).
    /// Only the landmarks FormAI relies on, to keep the overlay clean.
    static let connections: [(Int, Int)] = [
        // shoulders / torso
        (leftShoulder, rightShoulder),
        (leftShoulder, leftHip),
        (rightShoulder, rightHip),
        (leftHip, rightHip),
        // left arm
        (leftShoulder, leftElbow),
        (leftElbow, leftWrist),
        // right arm
        (rightShoulder, rightElbow),
        (rightElbow, rightWrist),
        // left leg
        (leftHip, leftKnee),
        (leftKnee, leftAnkle),
        // right leg
        (rightHip, rightKnee),
        (rightKnee, rightAnkle),
    ]

    /// All landmark indices touched by the overlay (so we can skip undetected ones).
    static let drawnJoints: [Int] = [
        leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist,
        leftHip, rightHip, leftKnee, rightKnee, leftAnkle, rightAnkle,
    ]
}
