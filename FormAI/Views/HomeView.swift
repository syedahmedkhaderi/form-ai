//
//  HomeView.swift
//  FormAI
//

import SwiftUI

struct HomeView: View {
    @State private var exercise: Exercise = .squat
    @State private var arm: Arm = .right
    @State private var goldenMessage: String = ""
    @State private var stats = WorkoutHistoryStore.shared.stats()

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 22) {
                    header
                    exerciseCards

                    if exercise == .curl {
                        armPicker
                    }

                    NavigationLink {
                        WorkoutView(exercise: exercise, arm: arm)
                    } label: {
                        Label("Start workout", systemImage: "play.fill")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)

                    progressSection
                    healthSection
                    developerSection
                }
                .padding()
            }
            .navigationTitle("FormAI")
            .background(Color(.systemGroupedBackground))
        }
        .navigationViewStyle(.stack)
        .onAppear {
            stats = WorkoutHistoryStore.shared.stats()
        }
    }

    private var header: some View {
        VStack(spacing: 6) {
            Image(systemName: "figure.run.circle.fill")
                .font(.system(size: 56))
                .foregroundStyle(.tint)
            Text("Real-time form coaching")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 8)
    }

    private var exerciseCards: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Exercise").font(.headline)
            HStack(spacing: 12) {
                ForEach(Exercise.allCases) { ex in
                    Button {
                        exercise = ex
                    } label: {
                        VStack(spacing: 10) {
                            Image(systemName: ex.systemImage).font(.system(size: 34))
                            Text(ex.displayName).font(.subheadline.weight(.semibold))
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 22)
                        .background(
                            RoundedRectangle(cornerRadius: 16)
                                .fill(exercise == ex ? Color.accentColor.opacity(0.15) : Color(.secondarySystemGroupedBackground))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(exercise == ex ? Color.accentColor : .clear, lineWidth: 2)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var armPicker: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Working arm").font(.headline)
            Picker("Working arm", selection: $arm) {
                ForEach(Arm.allCases) { Text($0.displayName).tag($0) }
            }
            .pickerStyle(.segmented)
        }
    }

    private var progressSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Progress", systemImage: "chart.line.uptrend.xyaxis")
                .font(.headline)

            HStack(spacing: 12) {
                statPill(title: "Workouts", value: "\(stats.totalWorkouts)")
                statPill(title: "Streak", value: "\(stats.streakDays)d")
                statPill(title: "Best Curl", value: bestValue(for: .curl))
            }

            if let lastWorkout = stats.lastWorkout {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Last workout")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Text("\(lastWorkout.exercise.displayName) · \(lastWorkout.reps) reps · avg \(lastWorkout.averageScore)")
                        .font(.subheadline.weight(.semibold))
                    Text(lastWorkout.timestamp.formatted(date: .abbreviated, time: .shortened))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                Text("No local history yet. Finish one workout to unlock streaks and recent session stats.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 18).fill(Color(.secondarySystemGroupedBackground)))
    }

    private func statPill(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.title3.weight(.bold))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 14).fill(Color.white.opacity(0.75)))
    }

    private func bestValue(for exercise: Exercise) -> String {
        guard let value = stats.bestAverageByExercise[exercise] else { return "—" }
        return "\(value)"
    }

    private var healthSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Health Tips", systemImage: "cross.case.fill")
                .font(.headline)
            Text("Stop the set if you feel pain, the pose keeps dropping out, or you cannot keep your full body in frame.")
                .font(.subheadline)
            Text("For squats, use a slight side angle. For curls, keep the working arm visible and move slowly to help the coach catch bad form early.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 18).fill(Color(.secondarySystemGroupedBackground)))
    }

    private var developerSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Developer").font(.headline)
            Button {
                goldenMessage = PreprocessGoldenTest.run(for: exercise).message
            } label: {
                Label("Run preprocessing self-test", systemImage: "checkmark.seal")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.bordered)
            if !goldenMessage.isEmpty {
                Text(goldenMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 16).fill(Color(.tertiarySystemGroupedBackground)))
    }
}
