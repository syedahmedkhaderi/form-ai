//
//  HomeView.swift
//  FormAI
//
//  Home screen with the original setup flow restored: intro, exercise picker,
//  arm picker for curls, start button, developer self-test, and a restrained
//  health tips section at the bottom.
//

import SwiftUI

struct HomeView: View {
    @State private var exercise: Exercise = .squat
    @State private var arm: Arm = .right
    @State private var goldenMessage: String = ""

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
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

                    developerSection
                    healthTipsSection
                }
                .padding()
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    Text("")
                }
            }
            .background(Color(.systemGroupedBackground))
        }
        .navigationViewStyle(.stack)
    }

    private var header: some View {
        VStack(spacing: 20) {
            Text("FormAI")
                .font(.system(size: 44, weight: .black, design: .rounded))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 20)

            Image(systemName: "figure.run.circle.fill")
                .font(.system(size: 56))
                .foregroundStyle(.tint)
            Text("Real-time form coaching")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var exerciseCards: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Exercise")
                .font(.system(size: 22, weight: .bold))
            HStack(spacing: 12) {
                ForEach(Exercise.allCases) { ex in
                    Button {
                        exercise = ex
                    } label: {
                        VStack(spacing: 10) {
                            Image(systemName: ex.systemImage).font(.system(size: 34))
                            Text(ex.displayName).font(.system(size: 18, weight: .semibold))
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
            Text("Working arm")
                .font(.system(size: 22, weight: .bold))
            Picker("Working arm", selection: $arm) {
                ForEach(Arm.allCases) { Text($0.displayName).tag($0) }
            }
            .pickerStyle(.segmented)
        }
    }

    private var developerSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Developer")
                .font(.system(size: 22, weight: .bold))
            Button {
                goldenMessage = PreprocessGoldenTest.run(for: exercise).message
            } label: {
                Label("Run preprocessing self-test", systemImage: "checkmark.seal")
                    .font(.headline)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 8)
            }
            .buttonStyle(.bordered)

            if !goldenMessage.isEmpty {
                Text(goldenMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 18)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }

    private var healthTipsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Health Tips")
                .font(.system(size: 22, weight: .bold))
            tipRow("Stop immediately if you feel sharp pain or dizziness.")
            tipRow("Keep your full body in frame so the coach can track safely.")
            tipRow("Use slow, controlled reps instead of rushing for count.")
            Text(exercise == .squat ? "For squats, use a slight side angle and keep your chest lifted." : "For curls, keep the working elbow close to your side and avoid swinging.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 18)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }

    private func tipRow(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Circle()
                .fill(Color.primary)
                .frame(width: 6, height: 6)
                .padding(.top, 7)
            Text(text)
                .font(.subheadline)
                .foregroundStyle(.primary)
        }
    }
}
