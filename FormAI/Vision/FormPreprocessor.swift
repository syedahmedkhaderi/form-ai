//
//  FormPreprocessor.swift
//  FormAI
//
//  THE contract-critical file (Islam doc section 6). This is a step-for-step
//  Swift port of the preprocessing recipe in 00_INTEGRATION_CONTRACTS.txt
//  section 7. The function "raw landmarks -> model input tensor" must be
//  byte-for-byte identical offline (Syed, Python) and on-device (here).
//
//  Verify with the golden test (PreprocessGoldenTest): feed Syed's known
//  input landmarks, confirm the output [32, F] matches within 1e-3. If that
//  passes, the model sees on-device exactly what it saw in training.
//
//  ─────────────────────────────────────────────────────────────────────────
//  CONVENTIONS that must match Syed's Python (documented because the contract
//  leaves the axis sign implicit):
//    • Image space: x,y in [0,1], y increases DOWNWARD (standard image coords).
//    • torso_lean: angle between the torso vector (sh_mid - hip_mid) and the
//      UP axis (0, -1). Upright torso -> ~0 degrees. Then /180.
//    • shoulder_tilt: angle between (working_shoulder - other_shoulder) and the
//      RIGHT axis (1, 0). Then /180.
//    • Curl uses a canonical "working side" representation: the selected arm is
//      mirrored into a right-arm frame, and the support side follows in the
//      feature vector. Reflection preserves interior angles, so elbow/knee
//      angles are unaffected; tilt/lean stay consistent because the sign is
//      applied to BOTH endpoints.
//  If Syed's reference differs on any of these, change it HERE and in his
//  model_card; nothing else couples.
//  ─────────────────────────────────────────────────────────────────────────
//

import Foundation

enum FormPreprocessor {

    /// Minimum torso scale; below this the person isn't well detected (contract).
    static let minScale: Float = 1e-4

    /// Up axis in image space (y grows downward, so "up" is -y).
    private static let upAxis = Vec2(x: 0, y: -1)
    /// Right axis in image space.
    private static let rightAxis = Vec2(x: 1, y: 0)

    /// Result of preprocessing one rep: a fixed [W=32, F] matrix, row-major.
    struct Window {
        let rows: Int          // W = 32
        let features: Int      // F
        /// `rows * features` values, row-major: index = r * features + c.
        let values: [Float]
    }

    /// Preprocess one rep.
    /// - Parameters:
    ///   - frames: the rep's frames (variable length T).
    ///   - exercise: squat or curl.
    ///   - arm: working arm (curl only; ignored for squat).
    /// - Returns: a [32, F] window, or nil if no usable frame remained.
    static func preprocess(frames: [PoseFrame], exercise: Exercise, arm: Arm = .right) -> Window? {
        let F = exercise.featureCount
        let W = exercise.windowLength

        // Steps 1 + 2: per-frame normalize, then build the ordered feature vector.
        var perFrame: [[Float]] = []
        perFrame.reserveCapacity(frames.count)
        for frame in frames {
            if let vec = featureVector(for: frame, exercise: exercise, arm: arm) {
                perFrame.append(vec)
            }
        }
        guard !perFrame.isEmpty else { return nil }

        // Step 3: resample the time axis from T (valid) to exactly W = 32.
        let resampled = resample(perFrame, to: W)
        let flat = resampled.flatMap { $0 }
        return Window(rows: W, features: F, values: flat)
    }

    // MARK: - Step 1: normalization

    /// Normalized landmarks (hip-center origin, torso-length scale) for the
    /// landmark indices `needed`. Returns nil if the frame must be dropped.
    private static func normalized(_ frame: PoseFrame, needed: [Int]) -> [Int: Vec2]? {
        let lh = frame[PoseLandmarkIndex.leftHip]
        let rh = frame[PoseLandmarkIndex.rightHip]
        let ls = frame[PoseLandmarkIndex.leftShoulder]
        let rs = frame[PoseLandmarkIndex.rightShoulder]

        // Required landmarks for normalization must be finite.
        for lm in [lh, rh, ls, rs] where !lm.x.isFinite || !lm.y.isFinite { return nil }

        let hipMid = Vec2(x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2)
        let shMid = Vec2(x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2)
        let scale = (shMid - hipMid).length
        guard scale >= minScale else { return nil }

        var out: [Int: Vec2] = [:]
        out.reserveCapacity(needed.count)
        for idx in needed {
            let p = frame[idx]
            guard p.x.isFinite, p.y.isFinite else { return nil }
            out[idx] = Vec2(x: (p.x - hipMid.x) / scale, y: (p.y - hipMid.y) / scale)
        }

        // Stash the normalized torso midpoints under sentinel keys for lean calc.
        out[normHipKey] = Vec2(x: 0, y: 0)  // hip is the origin after normalization
        out[normShKey] = Vec2(x: (shMid.x - hipMid.x) / scale, y: (shMid.y - hipMid.y) / scale)
        return out
    }

    private static let normHipKey = -1
    private static let normShKey = -2

    // MARK: - Step 2: ordered feature vector

    private static func featureVector(for frame: PoseFrame, exercise: Exercise, arm: Arm) -> [Float]? {
        switch exercise {
        case .squat: return squatFeatures(frame)
        case .curl:  return curlFeatures(frame, arm: arm)
        }
    }

    /// SQUAT, F = 21 (contract section 7).
    private static func squatFeatures(_ frame: PoseFrame) -> [Float]? {
        let I = PoseLandmarkIndex.self
        let needed = [I.leftShoulder, I.rightShoulder,
                      I.leftHip, I.rightHip,
                      I.leftKnee, I.rightKnee,
                      I.leftAnkle, I.rightAnkle]
        guard let n = normalized(frame, needed: needed) else { return nil }

        func p(_ i: Int) -> Vec2 { n[i]! }

        var f: [Float] = []
        f.reserveCapacity(21)

        // 16 normalized coords, in contract order: 11,12,23,24,25,26,27,28 (x,y each)
        for idx in [I.leftShoulder, I.rightShoulder, I.leftHip, I.rightHip,
                    I.leftKnee, I.rightKnee, I.leftAnkle, I.rightAnkle] {
            f.append(p(idx).x)
            f.append(p(idx).y)
        }

        // left_knee_angle = angle_at_B(23,25,27)/180
        f.append(Geometry.angleAtB(p(I.leftHip), p(I.leftKnee), p(I.leftAnkle)) / 180)
        // right_knee_angle = angle_at_B(24,26,28)/180
        f.append(Geometry.angleAtB(p(I.rightHip), p(I.rightKnee), p(I.rightAnkle)) / 180)
        // torso_lean = angle((sh_mid - hip_mid), up)/180
        let torso = n[normShKey]! - n[normHipKey]!
        f.append(Geometry.angleBetween(torso, axis: upAxis) / 180)
        // left_valgus = 25x - 27x  (knee_x - ankle_x, normalized)
        f.append(p(I.leftKnee).x - p(I.leftAnkle).x)
        // right_valgus = 26x - 28x
        f.append(p(I.rightKnee).x - p(I.rightAnkle).x)

        return f
    }

    /// CURL, F = 24. The working arm is canonical; left-arm reps are mirrored
    /// into the same right-arm frame used during training.
    private static func curlFeatures(_ frame: PoseFrame, arm: Arm) -> [Float]? {
        let I = PoseLandmarkIndex.self

        // Working-arm index mapping + x sign.
        let shoulder: Int, elbow: Int, wrist: Int, hip: Int
        let otherShoulder: Int, otherElbow: Int, otherWrist: Int, otherHip: Int
        let xSign: Float
        switch arm {
        case .right:
            shoulder = I.rightShoulder; elbow = I.rightElbow; wrist = I.rightWrist; hip = I.rightHip
            otherShoulder = I.leftShoulder; otherElbow = I.leftElbow; otherWrist = I.leftWrist; otherHip = I.leftHip
            xSign = 1
        case .left:
            shoulder = I.leftShoulder; elbow = I.leftElbow; wrist = I.leftWrist; hip = I.leftHip
            otherShoulder = I.rightShoulder; otherElbow = I.rightElbow; otherWrist = I.rightWrist; otherHip = I.rightHip
            xSign = -1
        }

        let needed = [I.nose, shoulder, elbow, wrist, hip, otherShoulder, otherElbow, otherWrist, otherHip]
        guard let raw = normalized(frame, needed: needed) else { return nil }

        // Apply the mirror sign to x for every used landmark.
        func p(_ i: Int) -> Vec2 { Vec2(x: xSign * raw[i]!.x, y: raw[i]!.y) }

        var f: [Float] = []
        f.reserveCapacity(24)

        // 18 normalized coords in canonical order.
        for idx in [shoulder, elbow, wrist, otherShoulder, otherElbow, otherWrist, hip, otherHip, I.nose] {
            f.append(p(idx).x)
            f.append(p(idx).y)
        }

        // working_elbow_angle = angle_at_B(shoulder, elbow, wrist)/180
        f.append(Geometry.angleAtB(p(shoulder), p(elbow), p(wrist)) / 180)
        // support_elbow_angle = angle_at_B(other_shoulder, other_elbow, other_wrist)/180
        f.append(Geometry.angleAtB(p(otherShoulder), p(otherElbow), p(otherWrist)) / 180)
        // shoulder_tilt = angle((working_shoulder - other_shoulder), right)/180
        let tilt = p(shoulder) - p(otherShoulder)
        f.append(Geometry.angleBetween(tilt, axis: rightAxis) / 180)
        // torso_lean = angle((sh_mid - hip_mid), up)/180
        let torso = raw[normShKey]! - raw[normHipKey]!
        f.append(Geometry.angleBetween(Vec2(x: xSign * torso.x, y: torso.y), axis: upAxis) / 180)
        // working_elbow_flare = elbow_x - shoulder_x
        f.append(p(elbow).x - p(shoulder).x)
        // support_elbow_flare = support_elbow_x - support_shoulder_x
        f.append(p(otherElbow).x - p(otherShoulder).x)

        return f
    }

    // MARK: - Step 3: linear-interpolate time axis T -> W

    private static func resample(_ frames: [[Float]], to W: Int) -> [[Float]] {
        let T = frames.count
        let F = frames[0].count
        if T == W { return frames }
        if T == 1 { return Array(repeating: frames[0], count: W) }

        var out: [[Float]] = []
        out.reserveCapacity(W)
        let denom = Float(W - 1)
        for i in 0..<W {
            let pos = Float(i) * Float(T - 1) / denom
            let i0 = Int(pos.rounded(.down))
            let i1 = min(i0 + 1, T - 1)
            let frac = pos - Float(i0)
            var row = [Float](repeating: 0, count: F)
            let a = frames[i0], b = frames[i1]
            for c in 0..<F {
                row[c] = a[c] + (b[c] - a[c]) * frac
            }
            out.append(row)
        }
        return out
    }
}
