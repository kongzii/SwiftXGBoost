import CXGBoost

/// C API: https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#a197d1b017fe9e62785b82397eb6bb17c
public class XGBoost {
    var features: [Feature]?
    let booster: UnsafeMutablePointer<BoosterHandle?>

    public static var systemLibraryVersion: Version {
        var major: Int32 = -1
        var minor: Int32 = -1
        var patch: Int32 = -1
        XGBoostVersion(&major, &minor, &patch)

        precondition(
            major >= 0 && minor >= 0 && patch >= 0,
            "XGBoostVersion does not return version."
        )

        return Version(
            major: Int(major),
            minor: Int(minor),
            patch: Int(patch)
        )
    }

    public static func registerLogCallback(
        _ call: (@convention(c) (UnsafePointer<Int8>?) -> Void)?
    ) throws {
        try safe {
            XGBRegisterLogCallback(call)
        }
    }

    public var pointee: BoosterHandle? {
        booster.pointee
    }

    public init(
        booster: UnsafeMutablePointer<BoosterHandle?>
    ) {
        self.booster = booster
    }

    public init(
        model: BufferModel
    ) throws {
        booster = .allocate(capacity: 1)

        try safe {
            XGBoosterUnserializeFromBuffer(
                booster,
                model.data,
                model.length
            )
        }
    }

    public init(
        with data: [Data] = [],
        from path: String? = nil,
        config: String? = nil,
        parameters: [Parameter] = [],
        validateParameters: Bool = true
    ) throws {
        booster = .allocate(capacity: 1)

        let pointees = data.map { $0.pointee }

        try safe {
            XGBoosterCreate(pointees, UInt64(pointees.count), booster)
        }

        if let path = path {
            try load(model: path)
        }

        if let config = config {
            try load(config: config)
        }

        if validateParameters {
            try set(parameter: "validate_parameters", value: "1")
        }

        for (name, value) in parameters {
            try set(parameter: name, value: value)
        }

        try validate(data: data)
    }

    deinit {
        try! safe {
            XGBoosterFree(pointee)
        }
    }

    public func config() throws -> String {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: 1)

        try safe {
            XGBoosterSaveJsonConfig(pointee, outLenght, outResult)
        }

        return String(cString: outResult.pointee!)
    }

    public func attributes() throws -> [String: String] {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafeMutablePointer<UnsafePointer<Int8>?>?>.allocate(capacity: 1)

        try safe {
            XGBoosterGetAttrNames(pointee, outLenght, outResult)
        }

        var attributes = [String: String]()
        let names = (0 ..< Int(outLenght.pointee)).lazy.map { String(cString: outResult.pointee![$0]!) }

        for name in names {
            attributes[name] = try attribute(name: name)!
        }

        return attributes
    }

    public func predict(
        from data: Data,
        outputMargin: Bool = false,
        treeLimit: UInt32 = 0,
        predictionLeaf: Bool = false,
        predictionContributions: Bool = false,
        approximateContributions: Bool = false,
        predictionInteractions: Bool = false,
        training: Bool = false,
        validateFeatures: Bool = true
    ) throws -> [Float] {
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
                pointee,
                data.pointee,
                optionMask,
                treeLimit,
                training ? 1 : 0,
                outLenght,
                outResult
            )
        }

        return (0 ..< Int(outLenght.pointee)).map { outResult.pointee![$0] }
    }

    public func predict(
        features: [Float],
        outputMargin: Bool = false,
        treeLimit: UInt32 = 0,
        predictionLeaf: Bool = false,
        predictionContributions: Bool = false,
        approximateContributions: Bool = false,
        predictionInteractions: Bool = false,
        training: Bool = false,
        missingValue: Float = Float.greatestFiniteMagnitude
    ) throws -> Float {
        try predict(
            from: Data(
                name: "predict",
                values: features,
                shape: (1, features.count),
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

    public func serialized() throws -> BufferModel {
        let length = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let data = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: 1)

        try safe {
            XGBoosterSerializeToBuffer(
                pointee,
                length,
                data
            )
        }

        return (
            length: length.pointee,
            data: data
        )
    }

    public func save(
        to path: String
    ) throws {
        try safe {
            XGBoosterSaveModel(pointee, path)
        }
    }

    public func raw() throws -> BufferModel {
        let length = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let data = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: 1)

        try safe {
            XGBoosterGetModelRaw(
                pointee,
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
        case .text:
            output = ""
            for (index, booster) in models.enumerated() {
                output += "booster[\(index)]:\n\(booster)"
            }
        }

        return output
    }

    public func dumped(
        features: String = "",
        withStatistics: Bool = false,
        format: ModelFormat = .text
    ) throws -> String {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafeMutablePointer<UnsafePointer<Int8>?>?>.allocate(capacity: 1)

        try safe {
            XGBoosterDumpModelEx(
                pointee,
                features,
                withStatistics ? 1 : 0,
                format.rawValue,
                outLenght,
                outResult
            )
        }

        let models = (0 ..< Int(outLenght.pointee)).map { String(cString: outResult.pointee![$0]!) }
        return formatModelDump(models: models, format: format)
    }

    public func dumped(
        features: [Feature],
        withStatistics: Bool = false,
        format: ModelFormat = .text
    ) throws -> String {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafeMutablePointer<UnsafePointer<Int8>?>?>.allocate(capacity: 1)

        var names = features.map { $0.name.cCompatible }
        var types = features.map { $0.type.rawValue.cCompatible }

        try safe {
            XGBoosterDumpModelExWithFeatures(
                pointee,
                Int32(names.count),
                &names,
                &types,
                withStatistics ? 1 : 0,
                format.rawValue,
                outLenght,
                outResult
            )
        }

        let models = (0 ..< Int(outLenght.pointee)).map { String(cString: outResult.pointee![$0]!) }
        return formatModelDump(models: models, format: format)
    }

    public func load(
        model buffer: BufferModel
    ) throws {
        try safe {
            XGBoosterLoadModelFromBuffer(
                pointee,
                buffer.data,
                buffer.length
            )
        }
    }

    public func load(
        model path: String
    ) throws {
        try safe {
            XGBoosterLoadModel(pointee, path)
        }
    }

    public func load(
        config: String
    ) throws {
        try safe {
            XGBoosterLoadJsonConfig(pointee, config)
        }
    }

    public func saveRabitCheckpoint() throws {
        try safe {
            XGBoosterSaveRabitCheckpoint(pointee)
        }
    }

    public func loadRabitCheckpoint() throws -> Int {
        var version: Int32 = -1
        try safe {
            XGBoosterLoadRabitCheckpoint(pointee, &version)
        }
        return Int(version)
    }

    public func attribute(
        name: String
    ) throws -> String? {
        var success: Int32 = -1
        let outResult = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: 1)

        try safe {
            XGBoosterGetAttr(pointee, name, outResult, &success)
        }

        if success != 1 {
            return nil
        }

        return String(cString: outResult.pointee!)
    }

    public func set(
        attribute: String,
        value: String
    ) throws {
        try safe {
            XGBoosterSetAttr(
                pointee, attribute, value
            )
        }
    }

    public func set(
        parameter: String,
        value: String
    ) throws {
        try safe {
            XGBoosterSetParam(pointee, parameter, value)
        }
    }

    public func update(
        iteration: Int,
        data: Data
    ) throws {
        try validate(data: data)

        try safe {
            XGBoosterUpdateOneIter(
                pointee,
                Int32(iteration),
                data.pointee
            )
        }
    }

    public func update(
        data: Data,
        objective: ([Float], Data) -> (gradient: [Float], hessian: [Float]),
        validateFeatures: Bool = true
    ) throws {
        let predicted = try predict(
            from: data,
            outputMargin: true,
            training: true,
            validateFeatures: validateFeatures
        )
        let (gradient, hessian) = objective(predicted, data)
        try boost(
            data: data,
            gradient: gradient,
            hessian: hessian,
            validateFeatures: false
        )
    }

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
                pointee,
                data.pointee,
                &gradient,
                &hessian,
                UInt64(gradient.count)
            )
        }
    }

    public func evaluate(
        iteration: Int,
        data: [Data]
    ) throws -> [String: [String: String]] {
        try validate(data: data)

        var pointees = data.map { $0.pointee }
        var names = data.map { $0.name.cCompatible }
        let output = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: 1)

        try safe {
            XGBoosterEvalOneIter(
                pointee,
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

            let name = nameSplitted[0]
            let metric = nameSplitted[1]
            let value = resultSplitted[1]

            if results[name] == nil {
                results[name] = [:]
            }

            results[name]![metric] = value
        }

        return results
    }

    public func evaluate(
        iteration: Int,
        data: Data
    ) throws -> [String: [String: String]] {
        try evaluate(iteration: iteration, data: [data])
    }

    public func train(
        iterations: Int,
        trainingData: Data,
        evaluationData: [Data] = [],
        beforeIteration: (XGBoost, Int) throws -> Void = { _, _ in },
        afterIteration: (XGBoost, Int, [String: [String: String]]?) throws -> AfterIterationCallbackOutput = { _, _, _ in .next }
    ) throws {
        training: for iteration in 0 ..< iterations {
            try beforeIteration(
                self,
                iteration
            )

            try update(
                iteration: iteration,
                data: trainingData
            )

            let evaluation =
                evaluationData.isEmpty ? nil : try evaluate(iteration: iteration, data: evaluationData)
            let output = try afterIteration(
                self,
                iteration,
                evaluation
            )

            switch output {
            case .stop:
                break training
            case .next:
                break
            }
        }
    }

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

    public func validate(
        data: Data
    ) throws {
        try validate(features: data.features())
    }

    public func validate(
        data: [Data]
    ) throws {
        for data in data {
            try validate(features: data.features())
        }
    }
}
