/// Type of feature.
public enum FeatureType: String {
    case quantitative = "q"
    case indicator = "i"
}

/// Information about feature.
public struct Feature: Equatable {
    public let name: String
    public let type: FeatureType

    public init(_ name: String, _ type: FeatureType) {
        self.name = name
        self.type = type
    }

    public init(name: String, type: FeatureType) {
        self.name = name
        self.type = type
    }
}
