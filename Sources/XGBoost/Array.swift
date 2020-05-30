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
