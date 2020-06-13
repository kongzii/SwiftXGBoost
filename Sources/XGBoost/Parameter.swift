/// Parameter for Booster.
/// See https://xgboost.readthedocs.io/en/latest/parameter.html for available parameters.
public struct Parameter: Equatable, Codable {
    var count: Int = 0

    public let name: String
    public let value: String

    public init(_ name: String, _ value: CustomStringConvertible) {
        self.name = name
        self.value = "\(value)"
    }

    public init(name: String, value: CustomStringConvertible) {
        self.name = name
        self.value = "\(value)"
    }
}
