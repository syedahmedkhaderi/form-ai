//
//  ExerciseConfig.swift
//  FormAI
//
//  Exercise/arm selection and the per-exercise constants from the
//  integration contract (W=32, F_squat=21, F_curl=24), plus the
//  label -> spoken cue mapping (Islam doc section 5f).
//

import Foundation

/// The two exercises in scope (contract: nothing else).
enum Exercise: String, CaseIterable, Identifiable {
    case squat
    case curl

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .squat: return "Squat"
        case .curl: return "Dumbbell Curl"
        }
    }

    var systemImage: String {
        switch self {
        case .squat: return "figure.strengthtraining.functional"
        case .curl: return "dumbbell.fill"
        }
    }

    /// Resampled window length W (contract: 32 for both).
    var windowLength: Int { 32 }

    /// Feature count F per frame (contract section 7).
    var featureCount: Int {
        switch self {
        case .squat: return 21
        case .curl: return 24
        }
    }

    /// Bundled Core ML model name (without extension). When Syed's
    /// `<name>.mlpackage` is added to the target, Xcode compiles it to
    /// `<name>.mlmodelc` which `FormScorer` loads by this name at runtime.
    var modelName: String {
        switch self {
        case .squat: return "SquatFormScorer"
        case .curl: return "CurlFormScorer"
        }
    }

    /// Bundled model card name (without extension).
    var modelCardResource: String {
        switch self {
        case .squat: return "model_card_squat"
        case .curl: return "model_card_curl"
        }
    }
}

/// Which arm is working (curl only). Squat ignores this.
enum Arm: String, CaseIterable, Identifiable {
    case right
    case left

    var id: String { rawValue }
    var displayName: String { self == .right ? "Right arm" : "Left arm" }
}

/// Maps a class label (from `model_card.class_labels`, or the rule-based
/// scorer) to a short spoken cue. Covers both v1 binary (good/bad) and the
/// v2 multi-class error types.
enum Cue {
    static func text(for label: String, exercise: Exercise) -> String {
        switch exercise {
        case .squat:
            switch label {
            case "good": return "Nice depth"
            case "knee_valgus": return "Push your knees out"
            case "insufficient_depth": return "Go lower"
            case "back_rounding": return "Chest up, flat back"
            case "bad": return "Fix your form"
            default: return "Keep going"
            }
        case .curl:
            switch label {
            case "good": return "Clean rep"
            case "swing": return "Stop swinging, control it"
            case "partial_rom": return "Full range, all the way up"
            case "elbow_flare": return "Tuck your elbow in"
            case "bad": return "Fix your form"
            default: return "Keep going"
            }
        }
    }

    /// Whether a label represents good form (drives score color / haptics).
    static func isGood(_ label: String) -> Bool { label == "good" }
}
