import CXGBoost

/// C API: https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html#a197d1b017fe9e62785b82397eb6bb17c
public class XGBoost {
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
        with data: [Data] = [],
        from path: String? = nil,
        config: String? = nil,
        parameters: [Parameter] = []
    ) throws {
        booster = .allocate(capacity: 1)

        let pointees = data.map { $0.pointee }

        try safe {
            XGBoosterCreate(pointees, UInt64(pointees.count), booster)
        }

        if let path = path {
            try loadModel(from: path)
        }

        if let config = config {
            try loadConfig(config: config)
        }

        for (name, value) in parameters {
            try setParameter(name: name, value: value)
        }
    }

    deinit {
        try! safe {
            XGBoosterFree(pointee)
        }
    }

    public func getConfig() throws -> String {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: 1)

        try safe {
            XGBoosterSaveJsonConfig(pointee, outLenght, outResult)
        }

        return String(cString: outResult.pointee!)
    }

    public func getAttributes() throws -> [String: String] {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafeMutablePointer<UnsafePointer<Int8>?>?>.allocate(capacity: 1)

        try safe {
            XGBoosterGetAttrNames(pointee, outLenght, outResult)
        }

        var attributes = [String: String]()
        let names = (0 ..< Int(outLenght.pointee)).lazy.map { String(cString: outResult.pointee![$0]!) }

        for name in names {
            attributes[name] = try getAttribute(name: name)!
        }

        return attributes
    }

    public func predict(
        from data: Data,
        mask: PredictOutputMask = .normal,
        treeLimit: UInt32 = 0,
        stage: Stage = .inference
    ) throws -> [Float] {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafePointer<Float>?>.allocate(capacity: 1)

        try safe {
            XGBoosterPredict(
                pointee,
                data.pointee,
                mask.rawValue,
                treeLimit,
                stage.rawValue,
                outLenght,
                outResult
            )
        }

        return (0 ..< Int(outLenght.pointee)).map { outResult.pointee![$0] }
    }

    public func predict(
        features: [Float],
        mask: PredictOutputMask = .normal,
        treeLimit: UInt32 = 0,
        stage: Stage = .inference,
        missingValue: Float = Float.greatestFiniteMagnitude
    ) throws -> Float {
        try predict(
            from: Data(
                name: "predict",
                values: features,
                rowCount: 1,
                columnCount: features.count,
                missingValue: missingValue
            ),
            mask: mask,
            treeLimit: treeLimit,
            stage: stage
        )[0]
    }

    public func saveModel(
        to path: String
    ) throws {
        try safe {
            XGBoosterSaveModel(pointee, path)
        }
    }

    public func dumpModel(
        features: String,
        withStatistics: Bool = false,
        format: ModelFormat = .text
    ) throws -> [String] {
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

        return (0 ..< Int(outLenght.pointee)).lazy.map { String(cString: outResult.pointee![$0]!) }
    }

    public func dumpModel(
        features: [Feature],
        withStatistics: Bool = false,
        format: ModelFormat = .text
    ) throws -> [String] {
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

        return (0 ..< Int(outLenght.pointee)).lazy.map { String(cString: outResult.pointee![$0]!) }
    }

    public func loadModel(
        from path: String
    ) throws {
        try safe {
            XGBoosterLoadModel(pointee, path)
        }
    }

    public func loadConfig(
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

    public func getAttribute(
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

    public func setAttribute(
        name: String,
        value: String
    ) throws {
        try safe {
            XGBoosterSetAttr(
                pointee, name, value
            )
        }
    }

    public func setParameter(
        name: String,
        value: String
    ) throws {
        try safe {
            XGBoosterSetParam(pointee, name, value)
        }
    }

    public func updateOneIter(
        iteration: Int,
        data: Data
    ) throws {
        try safe {
            XGBoosterUpdateOneIter(
                pointee,
                Int32(iteration),
                data.pointee
            )
        }
    }

    public func evalOneIter(
        iteration: Int,
        data: [Data]
    ) throws -> [String: [String: String]] {
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

            try updateOneIter(
                iteration: iteration,
                data: trainingData
            )

            let evaluation =
                evaluationData.isEmpty ? nil : try evalOneIter(iteration: iteration, data: evaluationData)
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
}
