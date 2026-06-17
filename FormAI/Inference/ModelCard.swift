//
//  ModelCard.swift
//  FormAI
//
//  Loads the bundled model_card.json. The class_labels list maps argmax
//  indices to labels; feature_order documents what the preprocessor produces.
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

    /// Load the bundled model card for an exercise, or a hardcoded default if none is bundled yet.
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
                exercise: "curl", W: 32, F: 24,
                feature_order: [],
                class_labels: ["good", "swing"],
                normalization: "hip-center origin, torso-length scale",
                version: "placeholder", built: nil)
        }
    }
}
