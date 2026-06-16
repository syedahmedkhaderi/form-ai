//
//  WorkoutView.swift
//  FormAI
//
//  The live workout screen: camera + skeleton overlay + framing guide +
//  big rep counter + form score + last spoken cue + model/rules toggle
//  (Islam doc section 8).
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

            SkeletonOverlay(frame: vm.currentFrame, imageSize: vm.imageSize, color: scoreColor)
                .ignoresSafeArea()

            FramingGuideOverlay(exercise: vm.exercise, visible: vm.currentFrame == nil && vm.isRunning)
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
                Button {
                    vm.flipCamera()
                } label: {
                    circleIcon("arrow.triangle.2.circlepath.camera")
                }
                .accessibilityLabel(vm.usingFrontCamera ? "Switch to back camera" : "Switch to front camera")
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
            // Last spoken cue
            HStack(spacing: 12) {
                Image(systemName: Cue.isGood(vm.lastLabel) ? "checkmark.circle.fill" : "waveform")
                    .font(.title2)
                    .foregroundStyle(Cue.isGood(vm.lastLabel) ? .green : .white)
                Text(vm.lastCue)
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.white)
                Spacer(minLength: 0)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 16))

            // Controls
            HStack(spacing: 12) {
                Toggle(isOn: $vm.useModel) {
                    Label(vm.useModel ? "AI model" : "Rules", systemImage: vm.useModel ? "brain" : "ruler")
                        .font(.subheadline.weight(.semibold))
                }
                .toggleStyle(.button)
                .tint(.white)
                .disabled(!vm.modelLoaded)

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
        }
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
}
