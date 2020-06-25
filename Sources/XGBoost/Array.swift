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
    /// - Returns: self
    public func data() throws -> [Float] {
        self
    }
}

extension Array: Int32Data where Element == Int32 {
    /// - Returns: self
    public func data() throws -> [Int32] {
        self
    }
}

extension Array: UInt32Data where Element == UInt32 {
    /// - Returns: self
    public func data() throws -> [UInt32] {
        self
    }
}

public extension Array {
    /// - Parameter into: Number of chunks.
    /// - Returns: Self splitted into n equally sized chunks.
    func chunked(into chunks: Int) -> [[Element]] {
        let size = (count / chunks)
            + Int((Float(count % chunks) / Float(chunks)).rounded(.up))

        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}

public extension Array where Element: AdditiveArithmetic {
    /// - Returns: Array calucated from self as [self[i + 1] - self[i]] for i = 0 ..< count - 1.
    func diff() -> [Element] {
        var result = [Element]()

        for i in 0 ..< count - 1 {
            result.append(self[i + 1] - self[i])
        }

        return result
    }
}

public extension Array where Element: FloatingPoint {
    /// - Parameter from: Initial value for sum.
    /// - Returns: Sum of elements.
    func sum(
        from initialValue: Element = 0
    ) -> Element {
        reduce(initialValue, +)
    }

    /// - Parameter sum: Precalculated sum.
    /// - Returns: Mean of elements.
    func mean(
        sum: Element? = nil
    ) -> Element {
        sum ?? self.sum() / Element(count)
    }

    /// - Parameter sum: Precalculated sum.
    /// - Parameter mean: Precalculated mean.
    /// - Parameter ddof: DDOF.
    /// - Returns: STD of elements.
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
    /// - Returns: Whether array contains `AfterIterationOutput.stop`.
    var willStop: Bool {
        contains(where: { $0 == .stop })
    }
}

public struct ArrayWithShape<Element>: Equatable, Sequence, ShapeData where Element: Equatable {
    public var array: [Element]
    public var shape: Shape

    public var count: Int {
        array.count
    }

    public init(_ array: [Element], shape: Shape) {
        self.array = array
        self.shape = shape
    }

    public func dataShape() throws -> Shape {
        shape
    }

    public subscript(index: Int) -> Element {
        get {
            array[index]
        }
        set(newValue) {
            array[index] = newValue
        }
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.array == rhs.array && lhs.shape == rhs.shape
    }

    public static func == (lhs: [Element], rhs: Self) -> Bool {
        lhs == rhs.array
    }

    public static func == (lhs: Self, rhs: [Element]) -> Bool {
        lhs.array == rhs
    }

    public func makeIterator() -> ArrayWithShapeIterator<Element> {
        ArrayWithShapeIterator<Element>(self)
    }
}

public struct ArrayWithShapeIterator<Element>: IteratorProtocol where Element: Equatable {
    var index = 0
    let arrayWithShape: ArrayWithShape<Element>

    init(_ arrayWithShape: ArrayWithShape<Element>) {
        self.arrayWithShape = arrayWithShape
    }

    public mutating func next() -> Element? {
        if index == arrayWithShape.count {
            return nil
        }

        defer { index += 1 }
        return arrayWithShape[index]
    }
}

extension ArrayWithShape: FloatData where Element == Float {
    /// - Returns: [Float]
    public func data() throws -> [Float] {
        array
    }
}

extension ArrayWithShape: Int32Data where Element == Int32 {
    /// - Returns: [Int32]
    public func data() throws -> [Int32] {
        array
    }
}

extension ArrayWithShape: UInt32Data where Element == UInt32 {
    /// - Returns: [UInt32]
    public func data() throws -> [UInt32] {
        array
    }
}
