//
//  WorkoutView.swift
//  FormAI
//
//  The live workout screen: camera + skeleton overlay + framing guide +
//  big rep counter + final score + one coaching card for the MVP.
//

import SwiftUI

struct WorkoutView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var vm: WorkoutViewModel

    init(exercise: Exercise, arm: Arm) {
        _vm = StateObject(wrappedValue: WorkoutViewModel(exercise: exercise, arm: arm))
    }

    var body: some View {
        ZStack {
            CameraPreview(session: vm.session)
                .ignoresSafeArea()

            FramingGuideOverlay(exercise: vm.exercise, visible: !vm.hasPoseInFrame && vm.isRunning)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                topBar
                statusBanners
                Spacer()
                bottomPanel
            }
            .padding()

            if vm.cameraDenied {
                cameraDeniedView
            }
        }
        .navigationBarBackButtonHidden(true)
        .navigationBarHidden(true)
        .task { await vm.start() }
        .onDisappear { vm.stop() }
    }

    // MARK: - Top

    private var topBar: some View {
        HStack(alignment: .top) {
            metric(title: "REPS", value: "\(vm.repCount)")
            Spacer()
            scoreMetric
            Spacer()
            VStack(spacing: 10) {
                Button {
                    vm.stop()
                    dismiss()
                } label: {
                    circleIcon("xmark")
                }
            }
        }
    }

    private func circleIcon(_ name: String) -> some View {
        Image(systemName: name)
            .font(.headline.bold())
            .foregroundStyle(.white)
            .frame(width: 40, height: 40)
            .background(.black.opacity(0.45), in: Circle())
    }

    private func metric(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption2.weight(.bold)).foregroundStyle(.white.opacity(0.8))
            Text(value).font(.system(size: 44, weight: .heavy, design: .rounded)).foregroundStyle(.white)
        }
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(.black.opacity(0.4), in: RoundedRectangle(cornerRadius: 14))
    }

    private var scoreMetric: some View {
        VStack(spacing: 2) {
            Text("FORM SCORE").font(.caption2.weight(.bold)).foregroundStyle(.white.opacity(0.8))
            Text(vm.repCount == 0 ? "—" : "\(vm.formScore)")
                .font(.system(size: 44, weight: .heavy, design: .rounded))
                .foregroundStyle(scoreColor)
        }
        .padding(.horizontal, 16).padding(.vertical, 8)
        .background(.black.opacity(0.4), in: RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - Banners

    @ViewBuilder private var statusBanners: some View {
        VStack(spacing: 6) {
            if !vm.poseAvailable {
                banner(icon: "exclamationmark.triangle.fill",
                       text: "Pose engine not linked — camera only. " + vm.poseStatus,
                       tint: .orange)
            }
            if !vm.modelLoaded {
                banner(icon: "cpu",
                       text: "No AI model yet — using rule-based scoring.",
                       tint: .blue)
            }
        }
        .padding(.top, 8)
    }

    private func banner(icon: String, text: String, tint: Color) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
            Text(text).font(.caption.weight(.medium))
            Spacer(minLength: 0)
        }
        .foregroundStyle(.white)
        .padding(10)
        .background(tint.opacity(0.85), in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Bottom

    private var bottomPanel: some View {
        VStack(spacing: 14) {
            coachingCard

            HStack(spacing: 12) {
                Button {
                    vm.voiceEnabled.toggle()
                } label: {
                    Label(vm.voiceEnabled ? "Voice on" : "Voice off",
                          systemImage: vm.voiceEnabled ? "speaker.wave.2.fill" : "speaker.slash.fill")
                        .font(.subheadline.weight(.semibold))
                }
                .buttonStyle(.bordered)
                .tint(.white)

                Spacer()

                if vm.exercise == .curl {
                    Text(vm.arm.displayName)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white.opacity(0.9))
                }
            }
            .padding(10)
            .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 16))

            if vm.liveState == .outOfFrame || vm.liveState == .stabilizing {
                Text(vm.exercise == .squat ? "Tip: use a slight side angle and keep your full body visible." : "Tip: keep the working arm, shoulder, and torso clearly visible.")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.white.opacity(0.85))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 6)
            }
        }
    }

    private var coachingCard: some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: coachingIcon)
                .font(.title2.weight(.bold))
                .foregroundStyle(vm.liveSeverity.color)
            VStack(alignment: .leading, spacing: 4) {
                Text(vm.liveState.title)
                    .font(.headline)
                    .foregroundStyle(.white)
                Text(vm.liveMessage)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.white.opacity(0.92))
            }
            Spacer(minLength: 0)
            Text(vm.liveSeverity.chipText)
                .font(.caption.weight(.bold))
                .foregroundStyle(vm.liveSeverity == .neutral ? .white : .black)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    Capsule()
                        .fill(vm.liveSeverity == .neutral ? Color.black.opacity(0.35) : vm.liveSeverity.color)
                )
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(coachingCardBackground, in: RoundedRectangle(cornerRadius: 16))
    }

    private var cameraDeniedView: some View {
        VStack(spacing: 12) {
            Image(systemName: "video.slash.fill").font(.largeTitle)
            Text("Camera access needed").font(.headline)
            Text("Enable camera access for FormAI in Settings to analyze your form.")
                .font(.subheadline).multilineTextAlignment(.center)
            Button("Back") { dismiss() }.buttonStyle(.borderedProminent)
        }
        .foregroundStyle(.white)
        .padding(24)
        .background(.black.opacity(0.85), in: RoundedRectangle(cornerRadius: 20))
        .padding(40)
    }

    private var scoreColor: Color {
        guard vm.repCount > 0 else { return .white }
        switch vm.formScore {
        case 80...: return .green
        case 50..<80: return .yellow
        default: return .red
        }
    }

    private var coachingIcon: String {
        switch vm.liveState {
        case .outOfFrame: return "viewfinder"
        case .stabilizing: return "pause.circle.fill"
        case .ready: return "checkmark.circle"
        case .repActive: return "figure.strengthtraining.traditional"
        case .warning: return "exclamationmark.triangle.fill"
        case .goodMovement: return "hand.thumbsup.fill"
        case .repComplete: return "number.circle.fill"
        }
    }

    private var coachingCardBackground: LinearGradient {
        let colors: [Color]
        switch vm.liveSeverity {
        case .neutral:
            colors = [Color.black.opacity(0.62), Color.black.opacity(0.42)]
        case .success:
            colors = [Color.green.opacity(0.82), Color.black.opacity(0.45)]
        case .caution:
            colors = [Color.yellow.opacity(0.78), Color.black.opacity(0.5)]
        case .warning:
            colors = [Color.orange.opacity(0.85), Color.red.opacity(0.55)]
        }
        return LinearGradient(colors: colors, startPoint: .leading, endPoint: .trailing)
    }
}
