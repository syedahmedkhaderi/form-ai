//
//  VoiceCoach.swift
//  FormAI
//
//  On-device text-to-speech coaching via AVSpeechSynthesizer.
//

import Foundation
import AVFoundation

final class VoiceCoach {
    private let synthesizer = AVSpeechSynthesizer()
    private var lastSpoken: String = ""
    private var lastSpokenAt: Date = .distantPast

    /// Minimum gap between identical cues so we don't nag every rep.
    private let repeatCooldown: TimeInterval = 3.0

    var isEnabled = true

    init() {
        configureAudioSession()
    }

    private func configureAudioSession() {
        let session = AVAudioSession.sharedInstance()
        // Duck other audio (e.g. user's music) instead of stopping it.
        try? session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers, .mixWithOthers])
        try? session.setActive(true)
    }

    func speak(_ text: String, interrupt: Bool = false) {
        guard isEnabled, !text.isEmpty else { return }

        // Avoid repeating the same cue back-to-back within the cooldown.
        if text == lastSpoken, Date().timeIntervalSince(lastSpokenAt) < repeatCooldown {
            return
        }
        lastSpoken = text
        lastSpokenAt = Date()

        if interrupt, synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }

        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0
        synthesizer.speak(utterance)
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }
}
