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
}
