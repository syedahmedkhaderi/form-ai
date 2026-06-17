//
//  ExerciseConfig.swift
//  FormAI
//
//  Exercise/arm selection and the per-exercise constants from the
//  integration contract (W=32, F_squat=21, F_curl=24), plus the
//  label -> spoken cue mapping (Islam doc section 5f).
//

import Foundation
import SwiftUI

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

enum LiveCoachingState: Equatable {
    case outOfFrame
    case stabilizing
    case ready
    case repActive
    case warning(label: String)
    case goodMovement
    case repComplete

    var title: String {
        switch self {
        case .outOfFrame: return "Get In Frame"
        case .stabilizing: return "Hold Still"
        case .ready: return "Ready"
        case .repActive: return "Rep Started"
        case .warning: return "Bad Form Now"
        case .goodMovement: return "Good Movement"
        case .repComplete: return "Rep Counted"
        }
    }

    func message(fallback: String) -> String {
        switch self {
        case .outOfFrame:
            return "Step back and keep your full body visible."
        case .stabilizing:
            return "Hold still for a second so tracking locks in."
        case .ready:
            return "Pose is locked. Start the rep when ready."
        case .repActive:
            return "Keep moving with control."
        case .warning:
            return fallback
        case .goodMovement:
            return "Form looks stable. Keep it controlled."
        case .repComplete:
            return fallback
        }
    }
}

enum LiveCoachingSeverity: String, Codable, Equatable {
    case neutral
    case success
    case caution
    case warning

    var color: Color {
        switch self {
        case .neutral: return .white
        case .success: return .green
        case .caution: return .yellow
        case .warning: return .orange
        }
    }

    var chipText: String {
        switch self {
        case .neutral: return "Watching form"
        case .success: return "Live"
        case .caution: return "Adjusting"
        case .warning: return "Needs correction"
        }
    }
}

struct LiveRuleEvaluation: Equatable {
    let label: String
    let message: String
    let severity: LiveCoachingSeverity
    let stable: Bool
    let shouldSpeak: Bool
}

struct WorkoutSummary: Codable, Equatable, Identifiable {
    let id: UUID
    let timestamp: Date
    let exerciseRawValue: String
    let reps: Int
    let averageScore: Int
    let finalScore: Int

    init(id: UUID = UUID(), timestamp: Date = Date(), exercise: Exercise, reps: Int, averageScore: Int, finalScore: Int) {
        self.id = id
        self.timestamp = timestamp
        self.exerciseRawValue = exercise.rawValue
        self.reps = reps
        self.averageScore = averageScore
        self.finalScore = finalScore
    }

    var exercise: Exercise {
        Exercise(rawValue: exerciseRawValue) ?? .squat
    }
}

struct WorkoutStats: Equatable {
    let totalWorkouts: Int
    let streakDays: Int
    let lastWorkout: WorkoutSummary?
    let bestAverageByExercise: [Exercise: Int]
}

final class WorkoutHistoryStore {
    static let shared = WorkoutHistoryStore()

    private let key = "formai.workoutHistory.v1"
    private let defaults = UserDefaults.standard
    private let calendar = Calendar.current

    private init() {}

    func record(_ summary: WorkoutSummary) {
        var items = loadAll()
        items.append(summary)
        save(items)
    }

    func stats() -> WorkoutStats {
        let workouts = loadAll().sorted { $0.timestamp > $1.timestamp }
        var bestAverageByExercise: [Exercise: Int] = [:]
        for summary in workouts {
            bestAverageByExercise[summary.exercise] = max(bestAverageByExercise[summary.exercise] ?? 0, summary.averageScore)
        }
        return WorkoutStats(
            totalWorkouts: workouts.count,
            streakDays: streakDays(from: workouts),
            lastWorkout: workouts.first,
            bestAverageByExercise: bestAverageByExercise
        )
    }

    private func loadAll() -> [WorkoutSummary] {
        guard let data = defaults.data(forKey: key),
              let items = try? JSONDecoder().decode([WorkoutSummary].self, from: data) else {
            return []
        }
        return items
    }

    private func save(_ items: [WorkoutSummary]) {
        guard let data = try? JSONEncoder().encode(items) else { return }
        defaults.set(data, forKey: key)
    }

    private func streakDays(from workouts: [WorkoutSummary]) -> Int {
        guard !workouts.isEmpty else { return 0 }
        let uniqueDays = workouts
            .map { calendar.startOfDay(for: $0.timestamp) }
            .reduce(into: [Date]()) { days, day in
                if !days.contains(day) { days.append(day) }
            }
            .sorted(by: >)

        guard let firstDay = uniqueDays.first else { return 0 }
        var expectedDay = firstDay
        var streak = 0
        for day in uniqueDays {
            if calendar.isDate(day, inSameDayAs: expectedDay) {
                streak += 1
                guard let next = calendar.date(byAdding: .day, value: -1, to: expectedDay) else { break }
                expectedDay = next
            } else if day < expectedDay {
                break
            }
        }
        return streak
    }
}
