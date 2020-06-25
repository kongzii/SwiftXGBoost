/// Type of feature.
public enum FeatureType: String {
    case quantitative = "q"
    case indicator = "i"
}

/// Struct holding information about feature.
public struct Feature: Equatable {
    public let name: String
    public let type: FeatureType

    /// - Parameter: Name of feature.
    /// - Parameter: Type of feature.
    public init(_ name: String, _ type: FeatureType) {
        self.name = name
        self.type = type
    }

    /// - Parameter name: Name of feature.
    /// - Parameter type: Type of feature.
    public init(name: String, type: FeatureType) {
        self.name = name
        self.type = type
    }
}
