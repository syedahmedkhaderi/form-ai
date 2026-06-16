//
//  Geometry.swift
//  FormAI
//
//  Small 2D vector helpers and the angle convention used by both the
//  preprocessing recipe (contract section 7) and the rule-based rep counter
//  (section 8). Kept dependency-light and pure so the math is easy to verify
//  against Syed's Python.
//

import Foundation

/// A 2D point in (normalized) image space.
struct Vec2 {
    var x: Float
    var y: Float

    static func - (a: Vec2, b: Vec2) -> Vec2 { Vec2(x: a.x - b.x, y: a.y - b.y) }
    static func + (a: Vec2, b: Vec2) -> Vec2 { Vec2(x: a.x + b.x, y: a.y + b.y) }
    static func / (a: Vec2, s: Float) -> Vec2 { Vec2(x: a.x / s, y: a.y / s) }

    var length: Float { (x * x + y * y).squareRoot() }
    func dot(_ o: Vec2) -> Float { x * o.x + y * o.y }
}

enum Geometry {
    /// Interior angle ABC at vertex B, in **degrees** [0, 180].
    /// Matches contract `angle_at_B(A, B, C)`.
    static func angleAtB(_ a: Vec2, _ b: Vec2, _ c: Vec2) -> Float {
        let ba = a - b
        let bc = c - b
        let denom = ba.length * bc.length
        guard denom > 1e-8 else { return 0 }
        var cosine = ba.dot(bc) / denom
        cosine = min(1, max(-1, cosine))
        return acos(cosine) * 180 / .pi
    }

    /// Unsigned angle between vector `v` and a reference axis, in degrees [0, 180].
    static func angleBetween(_ v: Vec2, axis: Vec2) -> Float {
        let denom = v.length * axis.length
        guard denom > 1e-8 else { return 0 }
        var cosine = v.dot(axis) / denom
        cosine = min(1, max(-1, cosine))
        return acos(cosine) * 180 / .pi
    }
}
