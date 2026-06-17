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

    static func liveEvaluation(
        frames: [PoseFrame],
        exercise: Exercise,
        arm: Arm,
        progress: RepProgress
    ) -> LiveRuleEvaluation? {
        guard frames.count >= 6 else { return nil }
        switch exercise {
        case .squat:
            return liveSquatEvaluation(frames, progress: progress)
        case .curl:
            return liveCurlEvaluation(frames, arm: arm, progress: progress)
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

    // MARK: - Live Squat

    private static func liveSquatEvaluation(_ frames: [PoseFrame], progress: RepProgress) -> LiveRuleEvaluation? {
        let I = PoseLandmarkIndex.self
        var kneeAngles: [Float] = []
        var valgusSignals: [Float] = []
        var torsoLeans: [Float] = []

        for frame in frames {
            guard let leftKnee = angle(frame, I.leftHip, I.leftKnee, I.leftAnkle),
                  let rightKnee = angle(frame, I.rightHip, I.rightKnee, I.rightAnkle) else { continue }
            kneeAngles.append((leftKnee + rightKnee) / 2)

            let lk = frame[I.leftKnee], la = frame[I.leftAnkle]
            let rk = frame[I.rightKnee], ra = frame[I.rightAnkle]
            if [lk, la, rk, ra].allSatisfy({ $0.x.isFinite && $0.y.isFinite }) {
                valgusSignals.append(abs((lk.x - la.x) - (rk.x - ra.x)))
            }

            let ls = frame[I.leftShoulder], rs = frame[I.rightShoulder], lh = frame[I.leftHip], rh = frame[I.rightHip]
            if [ls, rs, lh, rh].allSatisfy({ $0.x.isFinite && $0.y.isFinite }) {
                let shoulders = Vec2(x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2)
                let hips = Vec2(x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2)
                torsoLeans.append(Geometry.angleBetween(shoulders - hips, axis: Vec2(x: 0, y: -1)))
            }
        }

        guard !kneeAngles.isEmpty else { return nil }

        let minKnee = kneeAngles.min() ?? 180
        let avgValgus = valgusSignals.isEmpty ? 0 : valgusSignals.reduce(0, +) / Float(valgusSignals.count)
        let maxTorso = torsoLeans.max() ?? 0

        if progress == .rising && minKnee > 120 {
            return LiveRuleEvaluation(label: "insufficient_depth",
                                      message: "Go a little lower before standing up.",
                                      severity: .caution,
                                      stable: true,
                                      shouldSpeak: true)
        }
        if progress == .descending || progress == .bottom, avgValgus > 0.18 {
            return LiveRuleEvaluation(label: "knee_valgus",
                                      message: "Push your knees out as you lower.",
                                      severity: .warning,
                                      stable: true,
                                      shouldSpeak: true)
        }
        if (progress == .descending || progress == .bottom) && maxTorso > 38 {
            return LiveRuleEvaluation(label: "back_rounding",
                                      message: "Keep your chest up and torso tall.",
                                      severity: .warning,
                                      stable: true,
                                      shouldSpeak: true)
        }
        if progress == .descending || progress == .bottom || progress == .rising {
            return LiveRuleEvaluation(label: "good",
                                      message: "Good movement. Stay controlled.",
                                      severity: .success,
                                      stable: true,
                                      shouldSpeak: false)
        }
        return nil
    }

    // MARK: - Live Curl

    private static func liveCurlEvaluation(_ frames: [PoseFrame], arm: Arm, progress: RepProgress) -> LiveRuleEvaluation? {
        let I = PoseLandmarkIndex.self
        let shoulder = arm == .right ? I.rightShoulder : I.leftShoulder
        let elbow = arm == .right ? I.rightElbow : I.leftElbow
        let wrist = arm == .right ? I.rightWrist : I.leftWrist
        let otherShoulder = arm == .right ? I.leftShoulder : I.rightShoulder
        let otherElbow = arm == .right ? I.leftElbow : I.rightElbow
        let otherWrist = arm == .right ? I.leftWrist : I.rightWrist
        let otherHip = arm == .right ? I.leftHip : I.rightHip
        let workingHip = arm == .right ? I.rightHip : I.leftHip

        var elbowAngles: [Float] = []
        var torsoLeans: [Float] = []
        var flares: [Float] = []

        for frame in frames {
            if let angle = angle(frame, shoulder, elbow, wrist) {
                elbowAngles.append(angle)
            }

            let s = frame[shoulder], os = frame[otherShoulder], h = frame[workingHip], oh = frame[otherHip]
            if [s, os, h, oh].allSatisfy({ $0.x.isFinite && $0.y.isFinite }) {
                let shoulders = Vec2(x: (s.x + os.x) / 2, y: (s.y + os.y) / 2)
                let hips = Vec2(x: (h.x + oh.x) / 2, y: (h.y + oh.y) / 2)
                torsoLeans.append(Geometry.angleBetween(shoulders - hips, axis: Vec2(x: 0, y: -1)))
            }

            let elbowPoint = frame[elbow]
            let shoulderPoint = frame[shoulder]
            if elbowPoint.x.isFinite && shoulderPoint.x.isFinite {
                flares.append(abs(elbowPoint.x - shoulderPoint.x))
            }

            _ = otherElbow
            _ = otherWrist
        }

        guard let minElbow = elbowAngles.min(), let maxElbow = elbowAngles.max() else { return nil }
        let elbowRange = maxElbow - minElbow
        let maxTorso = torsoLeans.max() ?? 0
        let maxFlare = flares.max() ?? 0

        if (progress == .descending || progress == .rising || progress == .bottom) && maxTorso > 20 {
            return LiveRuleEvaluation(label: "swing",
                                      message: "Keep your torso still. Stop swinging.",
                                      severity: .warning,
                                      stable: true,
                                      shouldSpeak: true)
        }
        if (progress == .descending || progress == .bottom) && maxFlare > 0.16 {
            return LiveRuleEvaluation(label: "elbow_flare",
                                      message: "Tuck your elbow closer to your side.",
                                      severity: .caution,
                                      stable: true,
                                      shouldSpeak: true)
        }
        if progress == .rising && elbowRange < 45 {
            return LiveRuleEvaluation(label: "partial_rom",
                                      message: "Curl higher before lowering the weight.",
                                      severity: .caution,
                                      stable: true,
                                      shouldSpeak: true)
        }
        if progress == .descending || progress == .bottom || progress == .rising {
            return LiveRuleEvaluation(label: "good",
                                      message: "Good control. Keep the rep smooth.",
                                      severity: .success,
                                      stable: true,
                                      shouldSpeak: false)
        }
        return nil
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
