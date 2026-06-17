//
//  FormScorer.swift
//  FormAI
//
//  Core ML inference for one exercise. Loads the model dynamically at runtime
//  so the project compiles with no model present. When a `.mlpackage` is added
//  to the target, Xcode compiles it to `.mlmodelc` and this loader picks it up.
//
//  Model I/O:
//    input  "keypoint_window" : Float32, shape (1, W, F)
//    output "form_logits"     : Float32, shape (1, NUM_CLASSES)
//  Softmax + argmax are applied here; the model ships raw logits.
//

import Foundation
import CoreML

struct ScoreResult {
    let label: String
    /// 0...100, derived from P(good).
    let score: Int
    /// Max softmax probability (model confidence in its pick).
    let confidence: Float
    let probabilities: [Float]
    /// True when produced by the rule-based fallback, not the model.
    let fromFallback: Bool
}

final class FormScorer {
    let exercise: Exercise
    let card: ModelCard

    private var model: MLModel?
    private(set) var loadMessage: String = ""

    private let inputName = "keypoint_window"
    private let outputName = "form_logits"

    var isLoaded: Bool { model != nil }

    init(exercise: Exercise) {
        self.exercise = exercise
        self.card = ModelCard.load(for: exercise)
        loadModel()
    }

    private func loadModel() {
        guard let url = Bundle.main.url(forResource: exercise.modelName, withExtension: "mlmodelc") else {
            loadMessage = "\(exercise.modelName).mlmodelc not in bundle — add \(exercise.modelName).mlpackage "
                + "to the target. Using rule-based fallback until then."
            return
        }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            model = try MLModel(contentsOf: url, configuration: config)
            loadMessage = "Loaded \(exercise.modelName) (\(card.class_labels.count) classes)."
        } catch {
            loadMessage = "Failed to load \(exercise.modelName): \(error.localizedDescription). Using fallback."
        }
    }

    /// Run the model on a preprocessed window. Returns nil if the model is not
    /// loaded or inference fails (caller falls back to rules).
    func score(window: FormPreprocessor.Window) -> ScoreResult? {
        guard let model else { return nil }
        guard let multiArray = makeMultiArray(window) else { return nil }

        do {
            let input = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: multiArray)])
            let out = try model.prediction(from: input)
            guard let logitsValue = out.featureValue(for: outputName)?.multiArrayValue else {
                return nil
            }
            let logits = readFloats(logitsValue)
            let probs = Self.softmax(logits)
            return makeResult(probs: probs, fromFallback: false)
        } catch {
            return nil
        }
    }

    /// Build a label/score result from a probability vector.
    func makeResult(probs: [Float], fromFallback: Bool) -> ScoreResult {
        let argmax = probs.indices.max(by: { probs[$0] < probs[$1] }) ?? 0
        let label = argmax < card.class_labels.count ? card.class_labels[argmax] : "class_\(argmax)"
        let goodIdx = card.class_labels.firstIndex(of: "good")
        let pGood = goodIdx.map { $0 < probs.count ? probs[$0] : 0 } ?? probs[safe: 0] ?? 0
        let score = Int((pGood * 100).rounded())
        return ScoreResult(label: label,
                           score: max(0, min(100, score)),
                           confidence: probs[safe: argmax] ?? 0,
                           probabilities: probs,
                           fromFallback: fromFallback)
    }

    // MARK: - MLMultiArray plumbing

    private func makeMultiArray(_ window: FormPreprocessor.Window) -> MLMultiArray? {
        let shape: [NSNumber] = [1, NSNumber(value: window.rows), NSNumber(value: window.features)]
        guard let arr = try? MLMultiArray(shape: shape, dataType: .float32) else { return nil }
        let ptr = arr.dataPointer.bindMemory(to: Float32.self, capacity: window.values.count)
        for i in 0..<window.values.count {
            ptr[i] = window.values[i]
        }
        return arr
    }

    private func readFloats(_ arr: MLMultiArray) -> [Float] {
        let n = arr.count
        var out = [Float](repeating: 0, count: n)
        switch arr.dataType {
        case .float32:
            let ptr = arr.dataPointer.bindMemory(to: Float32.self, capacity: n)
            for i in 0..<n { out[i] = ptr[i] }
        case .double:
            let ptr = arr.dataPointer.bindMemory(to: Double.self, capacity: n)
            for i in 0..<n { out[i] = Float(ptr[i]) }
        default:
            for i in 0..<n { out[i] = arr[i].floatValue }
        }
        return out
    }

    static func softmax(_ logits: [Float]) -> [Float] {
        guard let maxLogit = logits.max() else { return [] }
        let exps = logits.map { expf($0 - maxLogit) }
        let sum = exps.reduce(0, +)
        guard sum > 0 else { return Array(repeating: 1 / Float(logits.count), count: logits.count) }
        return exps.map { $0 / sum }
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
