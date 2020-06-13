import Foundation

public extension Array where Element == Feature {
    /// Save feature map compatible with XGBoost`s inputs.
    ///
    /// - Parameter to: Path where feature map will be saved.
    func saveFeatureMap(to path: String) throws {
        try enumerated()
            .map { "\($0) \($1.name) \($1.type.rawValue)" }
            .joined(separator: "\n")
            .write(
                toFile: path,
                atomically: true,
                encoding: .utf8
            )
    }

    /// Load previously saved feature map.
    ///
    /// - Parameter from: Path to the feature map.
    init(fromFeatureMap path: String) throws {
        let content = try String(contentsOfFile: path)
        var features = [Feature]()

        for line in content.components(separatedBy: .newlines) {
            let splitted = line.components(separatedBy: " ")
            let (name, type) = (splitted[1], splitted[2])

            guard let featureType = FeatureType(rawValue: type) else {
                throw ValueError.runtimeError("Invalid feature type \(type).")
            }

            features.append(Feature(name: name, type: featureType))
        }

        self.init(features)
    }
}

extension Array: FloatData where Element == Float {
    public func data() throws -> [Float] {
        self
    }
}

extension Array: Int32Data where Element == Int32 {
    public func data() throws -> [Int32] {
        self
    }
}

extension Array: UInt32Data where Element == UInt32 {
    public func data() throws -> [UInt32] {
        self
    }
}

public extension Array {
    func chunked(into chunks: Int) -> [[Element]] {
        let size = (count / chunks)
            + Int((Float(count % chunks) / Float(chunks)).rounded(.up))

        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}

public extension Array where Element: AdditiveArithmetic {
    func diff() -> [Element] {
        var result = [Element]()

        for i in 0 ..< count - 1 {
            result.append(self[i + 1] - self[i])
        }

        return result
    }
}

public extension Array where Element: FloatingPoint {
    func sum() -> Element {
        reduce(0, +)
    }

    func mean(
        sum: Element? = nil
    ) -> Element {
        sum ?? self.sum() / Element(count)
    }

    func std(
        sum: Element? = nil,
        mean: Element? = nil,
        ddof: Int = 0
    ) -> Element {
        let mean = mean ?? self.mean(sum: sum)
        let v = reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
        return (v / Element(count - ddof)).squareRoot()
    }
}

public extension Array where Element == AfterIterationOutput {
    var willStop: Bool {
        contains(where: { $0 == .stop })
    }
}
