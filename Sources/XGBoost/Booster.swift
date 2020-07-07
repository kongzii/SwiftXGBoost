import CXGBoost
import Foundation

/// Alias for backward compatibility.
public typealias XGBoost = Booster

/// Typealias for objective function.
public typealias ObjectiveFunction = (ArrayWithShape<Float>, Data) throws -> (gradient: [Float], hessian: [Float])

/// Typealias for evaluation function.
public typealias EvaluationFunction = (ArrayWithShape<Float>, Data) throws -> (String, String)

/// Booster model.
///
/// Encapsulates BoosterHandle, the model of xgboost, that contains low level routines for
/// training, prediction and evaluation.
public class Booster {
    var features: [Feature]?
    var type: BoosterType?

    /// Pointer to underlying BoosterHandle.
    public var booster: BoosterHandle?

    /// Version of underlying XGBoost system library.
    public static var systemLibraryVersion = Version(
        major: Int(XGBOOST_VER_MAJOR),
        minor: Int(XGBOOST_VER_MINOR),
        patch: Int(XGBOOST_VER_PATCH)
    )

    /// Register callback function for LOG(INFO) messages.
    ///
    /// - Parameter call: Function to be called with C-String as parameter. Use String(cString: $0!) co convert it into Swift string.
    public static func registerLogCallback(
        _ call: (@convention(c) (UnsafePointer<Int8>?) -> Void)?
    ) throws {
        try safe {
            XGBRegisterLogCallback(call)
        }
    }

    /// Initialize Booster with an existing BoosterHandle pointer.
    ///
    /// - Parameter booster: BoosterHandle pointer.
    public init(
        booster: BoosterHandle?
    ) {
        self.booster = booster
    }

    /// Initialize Booster from buffer.
    ///
    /// - Parameter buffer: Model serialized as buffer.
    public convenience init(
        buffer: BufferModel
    ) throws {
        var booster: BoosterHandle?
        try safe {
            XGBoosterUnserializeFromBuffer(
                &booster,
                buffer.data,
                buffer.length
            )
        }
        self.init(booster: booster)
    }

    /// Initialize new Booster.
    ///
    /// - Parameter with: Data that will be cached.
    /// - Parameter from: Loads model from path.
    /// - Parameter config: Loads model from config.
    /// - Parameter parameters: Array of parameters to be set.
    /// - Parameter validateParameters: If true, parameters will be valided. This basically adds parameter validate_parameters=1.
    public convenience init(
        with data: [Data] = [],
        from path: String? = nil,
        config: String? = nil,
        parameters: [Parameter] = [],
        validateParameters: Bool = true
    ) throws {
        let pointees = data.map { $0.dmatrix }

        var booster: BoosterHandle?
        try safe {
            XGBoosterCreate(pointees, UInt64(pointees.count), &booster)
        }
        self.init(booster: booster)

        if let path = path {
            try load(model: path)
        }

        if let config = config {
            try load(config: config)
        }

        if validateParameters {
            try set(parameter: "validate_parameters", value: "1")
        }

        for parameter in parameters {
            try set(parameter: parameter.name, value: parameter.value)

            if parameter.name == "booster" {
                type = BoosterType(rawValue: parameter.value)

                if type == nil {
                    throw ValueError.runtimeError("Unknown booster type \(parameter.value).")
                }
            }
        }

        try validate(data: data)
    }

    deinit {
        try! safe {
            XGBoosterFree(booster)
        }
    }

    /// Serializes and unserializes booster to reset state and free training memory.
    public func reset() throws {
        let snapshot = try serialized()

        try safe {
            XGBoosterFree(booster)
        }

        try safe {
            XGBoosterUnserializeFromBuffer(
                &booster,
                snapshot.data,
                snapshot.length
            )
        }
    }

    /// - Returns: Booster's internal configuration in a JSON string.
    public func config() throws -> String {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: 1)

        try safe {
            XGBoosterSaveJsonConfig(booster, outLenght, outResult)
        }

        return String(cString: outResult.pointee!)
    }

    /// - Returns: Attributes stored in the Booster as a dictionary.
    public func attributes() throws -> [String: String] {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafeMutablePointer<UnsafePointer<Int8>?>?>.allocate(capacity: 1)

        try safe {
            XGBoosterGetAttrNames(booster, outLenght, outResult)
        }

        var attributes = [String: String]()
        let names = (0 ..< Int(outLenght.pointee)).lazy.map { String(cString: outResult.pointee![$0]!) }

        for name in names {
            attributes[name] = try attribute(name: name)!
        }

        return attributes
    }

    /// Predict from data.
    ///
    /// - Parameter from: Data to predict from.
    /// - Parameter outputMargin: Whether to output the raw untransformed margin value.
    /// - Parameter treeLimit: Limit number of trees in the prediction. Zero means use all trees.
    /// - Parameter predictionLeaf: Each record indicating the predicted leaf index of each sample in each tree.
    /// - Parameter predictionContributions: Each record indicating the feature contributions (SHAP values) for that prediction.
    /// - Parameter approximateContributions: Approximate the contributions of each feature.
    /// - Parameter predictionInteractions: Indicate the SHAP interaction values for each pair of features.
    /// - Parameter training: Whether the prediction will be used for traning.
    /// - Parameter validateFeatures: Validate booster and data features.
    public func predict(
        from data: DMatrix,
        outputMargin: Bool = false,
        treeLimit: UInt32 = 0,
        predictionLeaf: Bool = false,
        predictionContributions: Bool = false,
        approximateContributions: Bool = false,
        predictionInteractions: Bool = false,
        training: Bool = false,
        validateFeatures: Bool = true
    ) throws -> ArrayWithShape<Float> {
        if validateFeatures {
            try validate(data: data)
        }

        var optionMask: Int32 = 0x00

        if outputMargin {
            optionMask |= 0x01
        }

        if predictionLeaf {
            optionMask |= 0x02
        }

        if predictionContributions {
            optionMask |= 0x04
        }

        if approximateContributions {
            optionMask |= 0x08
        }

        if predictionInteractions {
            optionMask |= 0x10
        }

        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafePointer<Float>?>.allocate(capacity: 1)

        try safe {
            XGBoosterPredict(
                booster,
                data.dmatrix,
                optionMask,
                treeLimit,
                training ? 1 : 0,
                outLenght,
                outResult
            )
        }

        let predictions = (0 ..< Int(outLenght.pointee)).map { outResult.pointee![$0] }
        let rowCount = try data.rowCount()
        let columnCount = try data.columnCount()
        var shape = Shape(rowCount)

        if predictions.count != rowCount, predictions.count % rowCount == 0 {
            let chunkSize = predictions.count / rowCount

            if predictionInteractions {
                let nGroup = chunkSize / ((columnCount + 1) * (columnCount + 1))

                if nGroup == 1 {
                    shape = Shape(rowCount, columnCount + 1, columnCount + 1)
                } else {
                    shape = Shape(rowCount, nGroup, columnCount + 1, columnCount + 1)
                }
            } else if predictionContributions {
                let nGroup = chunkSize / (columnCount + 1)

                if nGroup == 1 {
                    shape = Shape(rowCount, columnCount + 1)
                } else {
                    shape = Shape(rowCount, nGroup, columnCount + 1)
                }
            } else {
                shape = Shape(rowCount, chunkSize)
            }
        }

        return ArrayWithShape<Float>(predictions, shape: shape)
    }

    /// Predict directly from array of floats, will build Data structure automatically with one row and features.count features.
    ///
    /// - Parameter features: Features to base prediction at.
    /// - Parameter outputMargin: Whether to output the raw untransformed margin value.
    /// - Parameter treeLimit: Limit number of trees in the prediction. Zero means use all trees.
    /// - Parameter predictionLeaf: Each record indicating the predicted leaf index of each sample in each tree.
    /// - Parameter predictionContributions: Each record indicating the feature contributions (SHAP values) for that prediction.
    /// - Parameter approximateContributions: Approximate the contributions of each feature.
    /// - Parameter predictionInteractions: Indicate the SHAP interaction values for each pair of features.
    /// - Parameter training: Whether the prediction will be used for traning.
    /// - Parameter missingValue: Value in features representing missing values.
    public func predict(
        features: FloatData,
        outputMargin: Bool = false,
        treeLimit: UInt32 = 0,
        predictionLeaf: Bool = false,
        predictionContributions: Bool = false,
        approximateContributions: Bool = false,
        predictionInteractions: Bool = false,
        training: Bool = false,
        missingValue: Float = Float.greatestFiniteMagnitude
    ) throws -> Float {
        let features = try features.data()
        return try predict(
            from: DMatrix(
                name: "predict",
                from: features,
                shape: Shape(1, features.count),
                features: self.features,
                missingValue: missingValue
            ),
            outputMargin: outputMargin,
            treeLimit: treeLimit,
            predictionLeaf: predictionLeaf,
            predictionContributions: predictionContributions,
            approximateContributions: approximateContributions,
            predictionInteractions: predictionInteractions,
            training: training
        )[0]
    }

    /// - Returns: Everything states in buffer.
    public func serialized() throws -> SerializedBuffer {
        let length = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let data = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: 1)

        try safe {
            XGBoosterSerializeToBuffer(
                booster,
                length,
                data
            )
        }

        return (
            length: length.pointee,
            data: data
        )
    }

    /// Saves modes into file.
    ///
    /// - Parameter to: Path to output file.
    public func save(
        to path: String
    ) throws {
        try safe {
            XGBoosterSaveModel(booster, path)
        }
    }

    /// - Returns: Model as binary raw bytes.
    public func raw() throws -> RawModel {
        let length = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let data = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: 1)

        try safe {
            XGBoosterGetModelRaw(
                booster,
                length,
                data
            )
        }

        return (
            length: length.pointee,
            data: data
        )
    }

    func formatModelDump(
        models: [String],
        format: ModelFormat
    ) -> String {
        var output: String

        switch format {
        case .json:
            output = "[\n\(models.joined(separator: ",\n"))]\n"
        case .text, .dot:
            output = ""
            for (index, booster) in models.enumerated() {
                output += "booster[\(index)]:\n\(booster)"
            }
        }

        return output
    }

    /// Dump model into a string.
    ///
    /// - Parameter features: Array of features.
    /// - Parameter featureMap: Name of the file containing feature map.
    /// - Parameter withStatistics: Controls whether the split statistics are output.
    /// - Parameter format: Desired output format type.
    /// - Returns: Formated output into ModelFormat format.
    public func dumped(
        features: [Feature]? = nil,
        featureMap: String = "",
        withStatistics: Bool = false,
        format: ModelFormat = .text
    ) throws -> String {
        try formatModelDump(
            models: rawDumped(
                features: features,
                featureMap: featureMap,
                withStatistics: withStatistics,
                format: format
            ),
            format: format
        )
    }

    /// Dump model into an array of strings.
    /// In most cases you will want to use `dumped` method to get output in expected format.
    ///
    /// - Parameter features: Array of features, you can override ones stored in self.features with this.
    /// - Parameter featureMap: Name of the file containing feature map.
    /// - Parameter withStatistics: Controls whether the split statistics are output.
    /// - Parameter format: Desired output format type.
    /// - Returns: Raw output from XGBoosterDumpModelEx provided as array of strings.
    public func rawDumped(
        features: [Feature]? = nil,
        featureMap: String = "",
        withStatistics: Bool = false,
        format: ModelFormat = .text
    ) throws -> [String] {
        let features = features ?? self.features

        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafeMutablePointer<UnsafePointer<Int8>?>?>.allocate(capacity: 1)

        if let boosterFeatures = features, featureMap == "" {
            var names = boosterFeatures.map { $0.name.cCompatible }
            var types = boosterFeatures.map { $0.type.rawValue.cCompatible }

            try safe {
                XGBoosterDumpModelExWithFeatures(
                    booster,
                    Int32(names.count),
                    &names,
                    &types,
                    withStatistics ? 1 : 0,
                    format.rawValue,
                    outLenght,
                    outResult
                )
            }
        } else {
            if featureMap != "", !FileManager.default.fileExists(atPath: featureMap) {
                throw ValueError.runtimeError("File \(featureMap) does not exists.")
            }

            try safe {
                XGBoosterDumpModelEx(
                    booster,
                    featureMap,
                    withStatistics ? 1 : 0,
                    format.rawValue,
                    outLenght,
                    outResult
                )
            }
        }

        return (0 ..< Int(outLenght.pointee)).map { String(cString: outResult.pointee![$0]!) }
    }

    /// Get feature importance of each feature.
    ///
    /// - Parameter featureMap: Path to the feature map.
    /// - Parameter importance: Type of importance you want to compute.
    /// - Returns: Tuple of features and gains, in case importance = weight, gains will be nil.
    public func score(
        featureMap: String = "",
        importance: Importance = .weight
    ) throws -> (features: [String: Int], gains: [String: Float]?) {
        if type != nil, ![.gbtree, .dart].contains(type!) {
            throw ValueError.runtimeError("Feature importance not defined for \(type!).")
        }

        if importance == .weight {
            var fMap = [String: Int]()
            let trees = try rawDumped(
                featureMap: featureMap,
                withStatistics: false,
                format: .text
            )

            for tree in trees {
                let lines = tree.components(separatedBy: "\n")
                for line in lines {
                    let splitted = line.components(separatedBy: "[")

                    if splitted.count == 1 {
                        // Leaf node
                        continue
                    }

                    let fid = splitted[1]
                        .components(separatedBy: "]")[0]
                        .components(separatedBy: "<")[0]

                    fMap[fid] = (fMap[fid] ?? 0) + 1
                }
            }

            return (fMap, nil)
        }

        var importance = importance
        var averageOverSplits = true

        if importance == .totalGain {
            importance = .gain
            averageOverSplits = false
        }

        if importance == .totalCover {
            importance = .cover
            averageOverSplits = false
        }

        var fMap = [String: Int]()
        var gMap = [String: Float]()
        let importanceSeparator = importance.rawValue + "="
        let trees = try rawDumped(
            featureMap: featureMap,
            withStatistics: true,
            format: .text
        )

        for tree in trees {
            let lines = tree.components(separatedBy: "\n")
            for line in lines {
                let splitted = line.components(separatedBy: "[")

                if splitted.count == 1 {
                    // Leaf node
                    continue
                }

                let feature = splitted[1].components(separatedBy: "]")
                let featureName = feature[0].components(separatedBy: "<")[0]
                let gain = Float(
                    feature[1]
                        .components(separatedBy: importanceSeparator)[1]
                        .components(separatedBy: ",")[0]
                )!

                fMap[featureName] = (fMap[featureName] ?? 0) + 1
                gMap[featureName] = (gMap[featureName] ?? 0) + gain
            }
        }

        if averageOverSplits {
            for (featureName, value) in gMap {
                gMap[featureName] = value / Float(fMap[featureName]!)
            }
        }

        return (fMap, gMap)
    }

    /// Loads model from buffer.
    ///
    /// - Parameter model: Buffer to load from.
    public func load(
        modelBuffer buffer: BufferModel
    ) throws {
        try safe {
            XGBoosterLoadModelFromBuffer(
                booster,
                buffer.data,
                buffer.length
            )
        }
    }

    /// Loads model from file.
    ///
    /// - Parameter model: Path of file to load model from.
    public func load(
        model path: String
    ) throws {
        try safe {
            XGBoosterLoadModel(booster, path)
        }
    }

    /// Loads model from config.
    ///
    /// - Parameter model: Config to load from.
    public func load(
        config: String
    ) throws {
        try safe {
            XGBoosterLoadJsonConfig(booster, config)
        }
    }

    /// Save the current checkpoint to rabit.
    public func saveRabitCheckpoint() throws {
        try safe {
            XGBoosterSaveRabitCheckpoint(booster)
        }
    }

    /// Initialize the booster from rabit checkpoint.
    ///
    /// - Returns: The output version of the model.
    public func loadRabitCheckpoint() throws -> Int {
        var version: Int32 = -1
        try safe {
            XGBoosterLoadRabitCheckpoint(booster, &version)
        }
        return Int(version)
    }

    /// Get attribute string from the Booster.
    ///
    /// - Parameter name: Name of attribute to get.
    /// - Returns: Value of attribute or nil if not set.
    public func attribute(
        name: String
    ) throws -> String? {
        var success: Int32 = -1
        let outResult = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: 1)

        try safe {
            XGBoosterGetAttr(booster, name, outResult, &success)
        }

        if success != 1 {
            return nil
        }

        return String(cString: outResult.pointee!)
    }

    /// Set string attribute.
    ///
    /// - Parameter attribute: Name of attribute.
    /// - Parameter value: Value of attribute.
    public func set(
        attribute: String,
        value: String
    ) throws {
        try safe {
            XGBoosterSetAttr(
                booster, attribute, value
            )
        }
    }

    /// Set string parameter.
    ///
    /// - Parameter parameter: Name of parameter.
    /// - Parameter value: Value of parameter.
    public func set(
        parameter: String,
        value: String
    ) throws {
        try safe {
            XGBoosterSetParam(booster, parameter, value)
        }
    }

    /// Update for one iteration, with objective function calculated internally.
    ///
    /// - Parameter iteration: Current iteration number.
    /// - Parameter data: Training data.
    /// - Parameter validateFeatures: Whether to validate features.
    public func update(
        iteration: Int,
        data: Data,
        validateFeatures: Bool = true
    ) throws {
        if validateFeatures {
            try validate(data: data)
        }

        try safe {
            XGBoosterUpdateOneIter(
                booster,
                Int32(iteration),
                data.dmatrix
            )
        }
    }

    /// Update for one iteration with custom objective.
    ///
    /// - Parameter data: Training data.
    /// - Parameter objective: Objective function returning gradient and hessian.
    /// - Parameter validateFeatures: Whether to validate features.
    public func update(
        data: Data,
        objective: ObjectiveFunction,
        validateFeatures: Bool = true
    ) throws {
        let predicted = try predict(
            from: data,
            outputMargin: true,
            training: true,
            validateFeatures: validateFeatures
        )
        let (gradient, hessian) = try objective(predicted, data)
        try boost(
            data: data,
            gradient: gradient,
            hessian: hessian,
            validateFeatures: false
        )
    }

    /// Boost the booster for one iteration, with customized gradient statistics.
    ///
    /// - Parameter data: Training data.
    /// - Parameter gradient: The first order of gradient.
    /// - Parameter hessian: The second order of gradient.
    /// - Parameter validateFeatures: Whether to validate features.
    public func boost(
        data: Data,
        gradient: [Float],
        hessian: [Float],
        validateFeatures: Bool = true
    ) throws {
        if gradient.count != hessian.count {
            throw ValueError.runtimeError(
                "Gradient count \(gradient.count) != Hessian count \(hessian.count)."
            )
        }

        if validateFeatures {
            try validate(data: data)
        }

        var gradient = gradient
        var hessian = hessian

        try safe {
            XGBoosterBoostOneIter(
                booster,
                data.dmatrix,
                &gradient,
                &hessian,
                UInt64(gradient.count)
            )
        }
    }

    /// Evaluate array of data.
    ///
    /// - Parameter iteration: Current iteration.
    /// - Parameter data: Data to evaluate.
    /// - Parameter function: Custom function for evaluation.
    /// - Returns: Dictionary in format [data_name: [eval_name: eval_value, ...], ...]
    public func evaluate(
        iteration: Int,
        data: [Data],
        function: EvaluationFunction? = nil
    ) throws -> [String: [String: String]] {
        try validate(data: data)

        var pointees = data.map { $0.dmatrix }
        var names = data.map { $0.name.cCompatible }
        let output = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: 1)

        try safe {
            XGBoosterEvalOneIter(
                booster,
                Int32(iteration),
                &pointees,
                &names,
                UInt64(data.count),
                output
            )
        }

        var results = [String: [String: String]]()
        let outputString = String(cString: output.pointee!)

        for (index, result) in outputString.components(separatedBy: .whitespacesAndNewlines).enumerated() {
            if index == 0 {
                continue
            }

            let resultSplitted = result.components(separatedBy: ":")

            let nameSplitted = resultSplitted[0].components(separatedBy: "-")
            let name = nameSplitted[0 ..< nameSplitted.count - 1].joined(separator: "-")
            let metric = nameSplitted.last!

            let value = resultSplitted[1]

            if results[name] == nil {
                results[name] = [:]
            }

            results[name]![metric] = value
        }

        if let function = function {
            for data in data {
                let (name, result) = try function(
                    try predict(from: data, training: false),
                    data
                )

                results[data.name]![name] = result
            }
        }

        return results
    }

    /// Evaluate data.
    ///
    /// - Parameter iteration: Current iteration.
    /// - Parameter data: Data to evaluate.
    /// - Parameter function: Custom function for evaluation.
    /// - Returns: Dictionary in format [data_name: [eval_name: eval_value]]
    public func evaluate(
        iteration: Int,
        data: Data,
        function: EvaluationFunction? = nil
    ) throws -> [String: String] {
        try evaluate(iteration: iteration, data: [data], function: function)[data.name]!
    }

    /// Validate features.
    ///
    /// - Parameter features: Features to validate.
    public func validate(
        features: [Feature]
    ) throws {
        guard let boosterFeatures = self.features else {
            self.features = features
            return
        }

        let featureNames = features.map { $0.name }
        let boosterFeatureNames = boosterFeatures.map { $0.name }

        if featureNames != boosterFeatureNames {
            let dataMissing = Set(boosterFeatureNames).subtracting(Set(featureNames))
            let boosterMissing = Set(featureNames).subtracting(Set(boosterFeatureNames))

            throw ValueError.runtimeError("""
            Feature names mismatch.
            Missing in data: \(dataMissing).
            Missing in booster: \(boosterMissing).
            """)
        }
    }

    /// Validate features.
    ///
    /// - Parameter data: Data which features will be validated.
    public func validate(
        data: DMatrix
    ) throws {
        try validate(features: data.features())
    }

    /// Validate features.
    ///
    /// - Parameter data: Array of data which features will be validated.
    public func validate(
        data: [DMatrix]
    ) throws {
        for data in data {
            try validate(features: data.features())
        }
    }
}
