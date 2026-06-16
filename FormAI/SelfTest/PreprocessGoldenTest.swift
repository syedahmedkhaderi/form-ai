//
//  PreprocessGoldenTest.swift
//  FormAI
//
//  Cross-language validation gate (Islam doc section 6). Syed exports a few
//  preprocessed reps with known input landmarks and known output [32, F]; we
//  feed the same inputs through the Swift preprocessor and confirm we
//  reproduce the output within 1e-3. If this passes, the model behaves
//  on-device exactly as in Syed's testing.
//
//  Golden file format (bundle a `golden_squat.json` / `golden_curl.json`):
//    {
//      "exercise": "squat",            // or "curl"
//      "arm": "right",                 // optional, curl only ("right"/"left")
//      "reps": [
//        {
//          "input":    [ [ [x,y,z,vis] x33 ] xT ],   // raw landmarks per frame
//          "expected": [ [ f x F ] x32 ]             // preprocessed window
//        }
//      ]
//    }
//
//  If no golden file is bundled, runs a synthetic shape/finiteness check
//  (matching Syed's Python unit test) so the harness still reports something.
//

import Foundation

struct GoldenTestResult {
    var passed: Bool
    var message: String
}

enum PreprocessGoldenTest {

    static func run(for exercise: Exercise) -> GoldenTestResult {
        if let url = Bundle.main.url(forResource: exercise.modelCardGoldenResource, withExtension: "json"),
           let data = try? Data(contentsOf: url) {
            return runGolden(data: data, exercise: exercise)
        }
        return runSynthetic(exercise: exercise)
    }

    // MARK: - Golden comparison

    private struct GoldenFile: Decodable {
        let exercise: String
        let arm: String?
        let reps: [GoldenRep]
    }
    private struct GoldenRep: Decodable {
        let input: [[[Float]]]   // [T][33][4]
        let expected: [[Float]]  // [32][F]
    }

    private static func runGolden(data: Data, exercise: Exercise) -> GoldenTestResult {
        guard let file = try? JSONDecoder().decode(GoldenFile.self, from: data) else {
            return GoldenTestResult(passed: false, message: "Golden JSON for \(exercise.rawValue) failed to parse.")
        }
        let arm = Arm(rawValue: file.arm ?? "right") ?? .right
        let tol: Float = 1e-3
        var maxErr: Float = 0

        for (r, rep) in file.reps.enumerated() {
            let frames = rep.input.map { frameRows -> PoseFrame in
                let lms = frameRows.map { Landmark(x: $0[0], y: $0[1], z: $0[2], visibility: $0.count > 3 ? $0[3] : 1) }
                return PoseFrame(landmarks: lms)
            }
            guard let window = FormPreprocessor.preprocess(frames: frames, exercise: exercise, arm: arm) else {
                return GoldenTestResult(passed: false, message: "Rep \(r): preprocessing returned nil.")
            }
            let expectedFlat = rep.expected.flatMap { $0 }
            guard expectedFlat.count == window.values.count else {
                return GoldenTestResult(passed: false,
                    message: "Rep \(r): shape mismatch (\(window.values.count) vs expected \(expectedFlat.count)).")
            }
            for i in 0..<expectedFlat.count {
                maxErr = max(maxErr, abs(expectedFlat[i] - window.values[i]))
            }
        }

        let passed = maxErr <= tol
        return GoldenTestResult(
            passed: passed,
            message: passed
                ? "Golden test PASSED for \(exercise.displayName): max error \(format(maxErr)) ≤ 1e-3."
                : "Golden test FAILED for \(exercise.displayName): max error \(format(maxErr)) > 1e-3.")
    }

    // MARK: - Synthetic fallback

    private static func runSynthetic(exercise: Exercise) -> GoldenTestResult {
        let frames = (0..<18).map { t -> PoseFrame in
            syntheticFrame(t: t, exercise: exercise)
        }
        guard let window = FormPreprocessor.preprocess(frames: frames, exercise: exercise) else {
            return GoldenTestResult(passed: false, message: "Synthetic preprocessing returned nil for \(exercise.rawValue).")
        }
        let expectedCount = exercise.windowLength * exercise.featureCount
        let shapeOK = window.values.count == expectedCount && window.rows == 32 && window.features == exercise.featureCount
        let finite = window.values.allSatisfy { $0.isFinite }
        let passed = shapeOK && finite
        return GoldenTestResult(
            passed: passed,
            message: passed
                ? "No golden file bundled — synthetic check PASSED for \(exercise.displayName): "
                  + "shape [32, \(exercise.featureCount)], all finite. Add \(exercise.modelCardGoldenResource).json from Syed for the real gate."
                : "Synthetic check FAILED for \(exercise.displayName): shapeOK=\(shapeOK), finite=\(finite).")
    }

    /// A plausible moving body so angles/derived features exercise real code paths.
    private static func syntheticFrame(t: Int, exercise: Exercise) -> PoseFrame {
        let phase = Float(t) / 18.0
        let bend = sin(phase * .pi) * 0.15   // a little squat/curl motion
        var lms = [Landmark](repeating: .zero, count: 33)
        func set(_ i: Int, _ x: Float, _ y: Float) {
            lms[i] = Landmark(x: x, y: y, z: 0, visibility: 1)
        }
        // Rough upright skeleton in [0,1] image space.
        set(PoseLandmarkIndex.leftShoulder, 0.45, 0.30)
        set(PoseLandmarkIndex.rightShoulder, 0.55, 0.30)
        set(PoseLandmarkIndex.leftElbow, 0.42, 0.42 - bend)
        set(PoseLandmarkIndex.rightElbow, 0.58, 0.42 - bend)
        set(PoseLandmarkIndex.leftWrist, 0.42, 0.54 - bend * 2)
        set(PoseLandmarkIndex.rightWrist, 0.58, 0.54 - bend * 2)
        set(PoseLandmarkIndex.leftHip, 0.46, 0.55)
        set(PoseLandmarkIndex.rightHip, 0.54, 0.55)
        set(PoseLandmarkIndex.leftKnee, 0.46, 0.72 + bend)
        set(PoseLandmarkIndex.rightKnee, 0.54, 0.72 + bend)
        set(PoseLandmarkIndex.leftAnkle, 0.46, 0.90)
        set(PoseLandmarkIndex.rightAnkle, 0.54, 0.90)
        _ = exercise
        return PoseFrame(landmarks: lms)
    }

    private static func format(_ v: Float) -> String { String(format: "%.2e", v) }
}

private extension Exercise {
    var modelCardGoldenResource: String {
        switch self {
        case .squat: return "golden_squat"
        case .curl: return "golden_curl"
        }
    }
}
