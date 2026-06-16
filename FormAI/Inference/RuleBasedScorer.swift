//
//  RuleBasedScorer.swift
//  FormAI
//
//  Threshold-based fallback (Islam doc section 7). The rep counter already
//  computes the key joint angles, so we can drive cues directly from per-rep
//  statistics when the ML model is weak or absent. This keeps the demo
//  bulletproof: default to the model, flip to rules if needed, and the app
//  always shows live form correction.
//
//  Thresholds are starting points — tune on real clips.
//

import Foundation

enum RuleBasedScorer {

    static func score(frames: [PoseFrame], exercise: Exercise, arm: Arm) -> ScoreResult {
        switch exercise {
        case .squat: return scoreSquat(frames)
        case .curl:  return scoreCurl(frames, arm: arm)
        }
    }

    // MARK: - Squat

    private static func scoreSquat(_ frames: [PoseFrame]) -> ScoreResult {
        let I = PoseLandmarkIndex.self
        var minKnee: Float = .greatestFiniteMagnitude
        var deepestFrame: PoseFrame?

        for f in frames {
            guard let l = angle(f, I.leftHip, I.leftKnee, I.leftAnkle),
                  let r = angle(f, I.rightHip, I.rightKnee, I.rightAnkle) else { continue }
            let avg = (l + r) / 2
            if avg < minKnee { minKnee = avg; deepestFrame = f }
        }

        guard minKnee < .greatestFiniteMagnitude, let deep = deepestFrame else {
            return result(label: "bad", score: 0)
        }

        // Depth: deeper squat -> smaller knee angle. ~95 deg or below is solid.
        let insufficientDepth = minKnee > 110

        // Valgus: at the bottom, knees collapse toward the midline.
        let lk = deep[I.leftKnee], la = deep[I.leftAnkle]
        let rk = deep[I.rightKnee], ra = deep[I.rightAnkle]
        // Left leg is on the right side of a front-facing image (larger x) is
        // not guaranteed, so detect symmetric inward drift: both knees pulled
        // toward each other relative to their ankles.
        let leftIn = (lk.x - la.x)
        let rightIn = (rk.x - ra.x)
        let kneeValgus = (leftIn * rightIn < 0) && (abs(leftIn) + abs(rightIn) > 0.04)

        if insufficientDepth {
            // Map depth onto a score: 95deg->~90, 160deg->~20.
            let s = Int(max(10, min(90, 90 - (minKnee - 95) * 1.2)))
            return result(label: "insufficient_depth", score: s)
        }
        if kneeValgus {
            return result(label: "knee_valgus", score: 55)
        }
        return result(label: "good", score: 92)
    }

    // MARK: - Curl

    private static func scoreCurl(_ frames: [PoseFrame], arm: Arm) -> ScoreResult {
        let I = PoseLandmarkIndex.self
        let shoulder = arm == .right ? I.rightShoulder : I.leftShoulder
        let elbow = arm == .right ? I.rightElbow : I.leftElbow
        let wrist = arm == .right ? I.rightWrist : I.leftWrist

        var minElbow: Float = .greatestFiniteMagnitude
        var maxElbow: Float = -.greatestFiniteMagnitude
        var minShoulderY: Float = .greatestFiniteMagnitude
        var maxShoulderY: Float = -.greatestFiniteMagnitude
        var maxFlare: Float = 0
        var any = false

        for f in frames {
            guard let e = angle(f, shoulder, elbow, wrist) else { continue }
            any = true
            minElbow = min(minElbow, e)
            maxElbow = max(maxElbow, e)
            let sy = f[shoulder].y
            if sy.isFinite {
                minShoulderY = min(minShoulderY, sy)
                maxShoulderY = max(maxShoulderY, sy)
            }
            let flare = abs(f[elbow].x - f[shoulder].x)
            maxFlare = max(maxFlare, flare)
        }

        guard any else { return result(label: "bad", score: 0) }

        let elbowRange = maxElbow - minElbow          // full curl ~ 90..150+
        let shoulderTravel = maxShoulderY - minShoulderY // sway up/down (normalized image y)

        let partialROM = elbowRange < 80
        let swinging = shoulderTravel > 0.06
        let elbowFlare = maxFlare > 0.12

        // Pick the dominant fault.
        if swinging {
            return result(label: "swing", score: 50)
        }
        if partialROM {
            let s = Int(max(15, min(85, 15 + elbowRange)))  // bigger range -> higher score
            return result(label: "partial_rom", score: s)
        }
        if elbowFlare {
            return result(label: "elbow_flare", score: 60)
        }
        return result(label: "good", score: 90)
    }

    // MARK: - Helpers

    private static func result(label: String, score: Int) -> ScoreResult {
        ScoreResult(label: label,
                    score: max(0, min(100, score)),
                    confidence: 1,
                    probabilities: [],
                    fromFallback: true)
    }

    private static func angle(_ frame: PoseFrame, _ a: Int, _ b: Int, _ c: Int) -> Float? {
        let pa = frame[a], pb = frame[b], pc = frame[c]
        for p in [pa, pb, pc] where !p.x.isFinite || !p.y.isFinite { return nil }
        return Geometry.angleAtB(Vec2(x: pa.x, y: pa.y),
                                 Vec2(x: pb.x, y: pb.y),
                                 Vec2(x: pc.x, y: pc.y))
    }
}
