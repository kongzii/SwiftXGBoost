/// Struct representating parameter for Booster.
/// See https://xgboost.readthedocs.io/en/latest/parameter.html for available parameters.
public struct Parameter: Equatable, Codable {
    var count: Int = 0

    public let name: String
    public let value: String

    /// - Parameter: Name of parameter.
    /// - Parameter: Value for parameter, can be anything convertible to string.
    public init(_ name: String, _ value: CustomStringConvertible) {
        self.name = name
        self.value = "\(value)"
    }

    /// - Parameter name: Name of parameter.
    /// - Parameter value: Value for parameter, can be anything convertible to string.
    public init(name: String, value: CustomStringConvertible) {
        self.name = name
        self.value = "\(value)"
    }
}
