//
//  RepCounter.swift
//  FormAI
//
//  Deterministic, rule-based rep detection (contract section 8 / Islam doc
//  section 5c). It does two jobs: count reps, and segment frames into per-rep
//  windows that feed the model. It also computes the same joint angles the
//  fallback scorer uses.
//
//  Squat: track knee angle. DOWN < 100 deg, UP > 160 deg. Count on DOWN->UP.
//  Curl:  track elbow angle. FLEXED < 60 deg, EXTENDED > 150 deg. Count on
//         FLEXED->EXTENDED.
//  Both reduce to the same shape: count a rep when the tracked angle returns
//  above the high threshold after having dropped below the low threshold. The
//  rep clip spans from the previous top through the bottom back to the top.
//

import Foundation

enum RepPhase {
    case idle        // haven't found the first top yet
    case top         // at the top of the movement (extended / standing)
    case bottom      // below the low threshold
    case moving      // between thresholds
}

struct RepEvent {
    var count: Int
    var currentAngle: Float
    var phase: RepPhase
    /// Non-nil exactly on the frame a rep completes; the buffered rep clip.
    var completedRep: [PoseFrame]?
}

final class RepCounter {
    let exercise: Exercise
    let arm: Arm

    private let lowThreshold: Float   // DOWN / FLEXED below this
    private let highThreshold: Float  // UP / EXTENDED above this

    private(set) var count = 0
    private var started = false
    private var hasBeenLow = false
    private var buffer: [PoseFrame] = []

    /// Safety cap so idling at the top doesn't grow the buffer forever.
    private let maxBufferFrames = 600

    init(exercise: Exercise, arm: Arm = .right) {
        self.exercise = exercise
        self.arm = arm
        switch exercise {
        case .squat:
            lowThreshold = 100
            highThreshold = 160
        case .curl:
            lowThreshold = 60
            highThreshold = 150
        }
    }

    func reset() {
        count = 0
        started = false
        hasBeenLow = false
        buffer.removeAll(keepingCapacity: true)
    }

    /// Feed one frame. Returns the current state and, when a rep just finished,
    /// the buffered clip for scoring.
    func consume(_ frame: PoseFrame) -> RepEvent {
        guard let angle = trackedAngle(frame) else {
            // Bad detection this frame: report idle-ish state, change nothing.
            return RepEvent(count: count, currentAngle: .nan, phase: started ? .moving : .idle, completedRep: nil)
        }

        let phase: RepPhase = angle > highThreshold ? .top : (angle < lowThreshold ? .bottom : .moving)

        // Wait for the first "top" before we start buffering a rep.
        if !started {
            if angle > highThreshold {
                started = true
                hasBeenLow = false
                buffer = [frame]
            }
            return RepEvent(count: count, currentAngle: angle, phase: started ? .top : .idle, completedRep: nil)
        }

        buffer.append(frame)
        if buffer.count > maxBufferFrames && !hasBeenLow {
            // Trim a long idle-at-top buffer, keep the most recent frames.
            buffer = Array(buffer.suffix(maxBufferFrames / 2))
        }

        if angle < lowThreshold {
            hasBeenLow = true
        }

        if angle > highThreshold && hasBeenLow {
            // DOWN->UP (or FLEXED->EXTENDED): a rep completes.
            count += 1
            let rep = buffer
            buffer = [frame]   // next rep starts at this top frame
            hasBeenLow = false
            return RepEvent(count: count, currentAngle: angle, phase: .top, completedRep: rep)
        }

        return RepEvent(count: count, currentAngle: angle, phase: phase, completedRep: nil)
    }

    // MARK: - Tracked angle (raw landmarks; angle is scale/translation invariant)

    private func trackedAngle(_ frame: PoseFrame) -> Float? {
        let I = PoseLandmarkIndex.self
        switch exercise {
        case .squat:
            // left_knee_angle = angle(hip23, knee25, ankle27)
            return angle(frame, I.leftHip, I.leftKnee, I.leftAnkle)
        case .curl:
            // elbow angle of the selected arm
            switch arm {
            case .right: return angle(frame, I.rightShoulder, I.rightElbow, I.rightWrist)
            case .left:  return angle(frame, I.leftShoulder, I.leftElbow, I.leftWrist)
            }
        }
    }

    private func angle(_ frame: PoseFrame, _ a: Int, _ b: Int, _ c: Int) -> Float? {
        let pa = frame[a], pb = frame[b], pc = frame[c]
        for p in [pa, pb, pc] where !p.x.isFinite || !p.y.isFinite { return nil }
        // Require some visibility so we don't count phantom poses.
        for p in [pa, pb, pc] where p.visibility < 0.3 { return nil }
        return Geometry.angleAtB(Vec2(x: pa.x, y: pa.y),
                                 Vec2(x: pb.x, y: pb.y),
                                 Vec2(x: pc.x, y: pc.y))
    }
}
