//
//  ModelCard.swift
//  FormAI
//
//  Loads Syed's sidecar `model_card.json` (contract section 6). The
//  `class_labels` list is the contract that turns an argmax index into a
//  label; `feature_order` documents what the Swift preprocessor must produce
//  (we don't parse it at runtime, but we surface it for the golden check).
//

import Foundation

struct ModelCard: Decodable {
    let exercise: String
    let W: Int
    let F: Int
    let feature_order: [String]
    let class_labels: [String]
    let normalization: String?
    let version: String?
    let built: String?

    /// Load the bundled model card for an exercise, or a sensible default
    /// matching the contract examples if none is bundled yet.
    static func load(for exercise: Exercise) -> ModelCard {
        if let url = Bundle.main.url(forResource: exercise.modelCardResource, withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let card = try? JSONDecoder().decode(ModelCard.self, from: data) {
            return card
        }
        return .default(for: exercise)
    }

    static func `default`(for exercise: Exercise) -> ModelCard {
        switch exercise {
        case .squat:
            return ModelCard(
                exercise: "squat", W: 32, F: 21,
                feature_order: [],
                class_labels: ["good", "knee_valgus", "insufficient_depth", "back_rounding"],
                normalization: "hip-center origin, torso-length scale",
                version: "placeholder", built: nil)
        case .curl:
            return ModelCard(
                exercise: "curl", W: 32, F: 11,
                feature_order: [],
                class_labels: ["good", "swing", "partial_rom", "elbow_flare"],
                normalization: "hip-center origin, torso-length scale",
                version: "placeholder", built: nil)
        }
    }
}
