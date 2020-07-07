import PythonKit
import XCTest

@testable import XGBoost

private let PJSON = Python.import("json")

final class BoosterTests: XCTestCase {
    func testAttribute() throws {
        let booster = try Booster()
        try booster.set(attribute: "testName", value: "testValue")
        XCTAssertEqual(try booster.attribute(name: "testName"), "testValue")
        XCTAssertNil(try booster.attribute(name: "unknownName"))
    }

    func testAttributes() throws {
        let booster = try Booster()
        try booster.set(attribute: "attribute1", value: "value1")
        try booster.set(attribute: "attribute2", value: "value2")
        XCTAssertEqual(try booster.attributes(), ["attribute1": "value1", "attribute2": "value2"])
    }

    func testJsonDumped() throws {
        let randomArray = (0 ..< 10).map { _ in Float.random(in: 0 ..< 2) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let data = try DMatrix(
            name: "data",
            from: randomArray,
            shape: Shape(10, 1),
            label: label,
            threads: 1
        )
        let booster = try Booster(with: [data])
        try booster.train(
            iterations: 5,
            trainingData: data
        )

        let pyBooster = try python(booster: booster)

        let temporaryDumpFile = temporaryFile()
        let temporaryFeatureMapFile = temporaryFile()
        try booster.features!.saveFeatureMap(to: temporaryFeatureMapFile)

        pyBooster.dump_model(fout: temporaryDumpFile, fmap: temporaryFeatureMapFile, dump_format: "json")

        let json = try booster.dumped(format: .json)
        let pyJson = try String(contentsOfFile: temporaryDumpFile)

        let jsonObject = PJSON.loads(json)
        let pyJsonObject = PJSON.loads(pyJson)

        XCTAssertTrue(jsonObject == pyJsonObject)
    }

    func testTextDumped() throws {
        let randomArray = (0 ..< 10).map { _ in Float.random(in: 0 ..< 2) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let data = try DMatrix(
            name: "data",
            from: randomArray,
            shape: Shape(10, 1),
            label: label,
            threads: 1
        )
        let booster = try Booster(with: [data])
        try booster.train(
            iterations: 5,
            trainingData: data
        )

        let pyBooster = try python(booster: booster)

        let temporaryDumpFile = temporaryFile()
        let temporaryFeatureMapFile = temporaryFile()

        let features = [Feature(name: "x", type: .quantitative)]
        try features.saveFeatureMap(to: temporaryFeatureMapFile)

        pyBooster.dump_model(
            fout: temporaryDumpFile, fmap: temporaryFeatureMapFile,
            with_stats: true, dump_format: "text"
        )

        let textFeatures = try booster.dumped(
            features: features, withStatistics: true, format: .text
        )
        let textFeatureMap = try booster.dumped(
            featureMap: temporaryFeatureMapFile, withStatistics: true, format: .text
        )
        let pyText = try String(contentsOfFile: temporaryDumpFile)

        XCTAssertEqual(textFeatures, pyText)
        XCTAssertEqual(textFeatureMap, pyText)
    }

    func testDotDumped() throws {
        let randomArray = (0 ..< 10).map { _ in Float.random(in: 0 ..< 2) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let data = try DMatrix(
            name: "data",
            from: randomArray,
            shape: Shape(10, 1),
            label: label,
            threads: 1
        )
        let booster = try Booster(with: [data])
        try booster.train(
            iterations: 5,
            trainingData: data
        )

        let pyBooster = try python(booster: booster)

        let dot = try booster.rawDumped(format: .dot)
        let pyDot = pyBooster.get_dump(dump_format: "dot").map { String($0)! }
        XCTAssertEqual(dot, pyDot)

        let temporaryFeatureMapFile = temporaryFile()
        try [Feature(name: "x", type: .quantitative)].saveFeatureMap(to: temporaryFeatureMapFile)

        let formatDot = try booster.dumped(featureMap: temporaryFeatureMapFile, format: .dot)
        let temporaryDumpFile = temporaryFile()
        pyBooster.dump_model(temporaryDumpFile, fmap: temporaryFeatureMapFile, dump_format: "dot")
        let pyFormatDot = try String(contentsOfFile: temporaryDumpFile)
        XCTAssertEqual(formatDot, pyFormatDot)
    }

    func testScoreEmptyFeatureMap() throws {
        let randomArray = (0 ..< 50).map { _ in Float.random(in: 0 ..< 2) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let data = try DMatrix(
            name: "data",
            from: randomArray,
            shape: Shape(10, 5),
            label: label,
            threads: 1
        )
        let booster = try Booster(with: [data])
        try booster.train(
            iterations: 10,
            trainingData: data
        )

        let pyBooster = try python(booster: booster)

        let pyWeightMap = [String: Int](pyBooster.get_score(
            fmap: "", importance_type: "weight"
        ))!
        let (weightMap, _) = try booster.score(featureMap: "", importance: .weight)
        XCTAssertEqual(weightMap, pyWeightMap)

        let pyGainMap = [String: Float](pyBooster.get_score(
            fmap: "", importance_type: "gain"
        ))!
        let (_, gainMap) = try booster.score(featureMap: "", importance: .gain)
        assertEqual(gainMap!, pyGainMap, accuracy: 1e-6)

        let pyTotalGainMap = [String: Float](pyBooster.get_score(
            fmap: "", importance_type: "total_gain"
        ))!
        let (_, totalGainMap) = try booster.score(featureMap: "", importance: .totalGain)
        assertEqual(totalGainMap!, pyTotalGainMap, accuracy: 1e-6)

        let pyCoverMap = [String: Float](pyBooster.get_score(
            fmap: "", importance_type: "cover"
        ))!
        let (_, coverMap) = try booster.score(featureMap: "", importance: .cover)
        assertEqual(coverMap!, pyCoverMap, accuracy: 1e-6)

        let pyTotalCoverMap = [String: Float](pyBooster.get_score(
            fmap: "", importance_type: "total_cover"
        ))!
        let (_, totalCoverMap) = try booster.score(featureMap: "", importance: .totalCover)
        assertEqual(totalCoverMap!, pyTotalCoverMap, accuracy: 1e-6)
    }

    func testScoreWithFeatureMap() throws {
        let randomArray = (0 ..< 50).map { _ in Float.random(in: 0 ..< 2) }
        let features = (0 ..< 5).map { Feature(name: "Feature-\($0)", type: .quantitative) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let data = try DMatrix(
            name: "data",
            from: randomArray,
            shape: Shape(10, 5),
            features: features,
            label: label,
            threads: 1
        )
        let booster = try Booster(with: [data])
        try booster.train(
            iterations: 10,
            trainingData: data
        )

        let temporaryNamesFile = temporaryFile()

        try features.saveFeatureMap(to: temporaryNamesFile)

        let pyBooster = try python(booster: booster)

        let pyWeightMap = [String: Int](pyBooster.get_score(
            fmap: temporaryNamesFile, importance_type: "weight"
        ))!
        let (weightMap, _) = try booster.score(
            featureMap: temporaryNamesFile, importance: .weight
        )
        XCTAssertEqual(weightMap, pyWeightMap)

        let pyGainMap = [String: Float](pyBooster.get_score(
            fmap: temporaryNamesFile, importance_type: "gain"
        ))!
        let (_, gainMap) = try booster.score(
            featureMap: temporaryNamesFile, importance: .gain
        )
        assertEqual(gainMap!, pyGainMap, accuracy: 1e-6)

        let pyTotalGainMap = [String: Float](pyBooster.get_score(
            fmap: temporaryNamesFile, importance_type: "total_gain"
        ))!
        let (_, totalGainMap) = try booster.score(
            featureMap: temporaryNamesFile, importance: .totalGain
        )
        assertEqual(totalGainMap!, pyTotalGainMap, accuracy: 1e-6)

        let pyCoverMap = [String: Float](pyBooster.get_score(
            fmap: temporaryNamesFile, importance_type: "cover"
        ))!
        let (_, coverMap) = try booster.score(
            featureMap: temporaryNamesFile, importance: .cover
        )
        assertEqual(coverMap!, pyCoverMap, accuracy: 1e-6)

        let pyTotalCoverMap = [String: Float](pyBooster.get_score(
            fmap: temporaryNamesFile, importance_type: "total_cover"
        ))!
        let (_, totalCoverMap) = try booster.score(
            featureMap: temporaryNamesFile, importance: .totalCover
        )
        assertEqual(totalCoverMap!, pyTotalCoverMap, accuracy: 1e-6)
    }

    func testSystemLibraryVersion() throws {
        let version = XGBoost.systemLibraryVersion

        XCTAssertGreaterThanOrEqual(version.major, 1)
        XCTAssertGreaterThanOrEqual(version.minor, 1)
        XCTAssertGreaterThanOrEqual(version.patch, 0)
    }

    func testConfig() throws {
        let json = Python.import("json")

        let booster = try randomTrainedBooster()
        let pyBooster = try python(booster: booster)

        let config = json.loads(try booster.config())
        let pyConfig = json.loads(pyBooster.save_config())

        config["version"] = Python.None
        pyConfig["version"] = Python.None

        XCTAssertEqual(String(json.dumps(config))!, String(json.dumps(pyConfig))!)
    }

    func testRawDumped() throws {
        let booster = try randomTrainedBooster()
        let pyBooster = try python(booster: booster)

        let rawDump = try booster.rawDumped()
        let pyRawDump = [String](pyBooster.get_dump())!
        XCTAssertEqual(rawDump, pyRawDump)

        let rawDumpStatistics = try booster.rawDumped(withStatistics: true)
        let pyRawDumpStatistics = [String](pyBooster.get_dump(with_stats: true))!
        XCTAssertEqual(rawDumpStatistics, pyRawDumpStatistics)
    }

    func testLoadModel() throws {
        let booster = try randomTrainedBooster()
        try booster.set(attribute: "hello", value: "world")
        let modelFile = temporaryFile()

        try booster.save(to: modelFile)
        let boosterLoaded = try Booster(from: modelFile)

        XCTAssertEqual(try booster.attribute(name: "hello"), try boosterLoaded.attribute(name: "hello"))
    }

    func testLoadConfig() throws {
        let booster1 = try randomTrainedBooster()
        try booster1.set(parameter: "eta", value: "555")

        let modelFile = temporaryFile()
        try booster1.save(to: modelFile)

        let booster2 = try randomTrainedBooster()
        try booster2.set(parameter: "eta", value: "666")
        let config = try booster2.config()

        let booster3 = try Booster(from: modelFile, config: config)
        XCTAssertEqual(try booster3.config(), try booster2.config())
    }

    func testInitializeWithType() throws {
        XCTAssertEqual(try Booster(
            parameters: [Parameter("booster", "dart")]
        ).type, BoosterType(rawValue: "dart"))
        XCTAssertThrowsError(try Booster(
            parameters: [Parameter("booster", "icecream")]
        ))
    }

    func testBoosterPredictOptionMasks() throws {
        let booster = try randomTrainedBooster()
        let pyBooster = try python(booster: booster)

        let test = try randomDMatrix()
        let pyTest = try python(dmatrix: test)

        let outputMargin = try booster.predict(from: test, outputMargin: true)
        let pyoutputMargin = pyBooster.predict(data: pyTest, output_margin: true)
        XCTAssertEqual(try outputMargin.data(), try pyoutputMargin.data())
        XCTAssertEqual(try outputMargin.dataShape(), try pyoutputMargin.dataShape())

        let predictionleaf = try booster.predict(from: test, predictionLeaf: true)
        let pypredictionleaf = pyBooster.predict(data: pyTest, pred_leaf: true)
        XCTAssertEqual(try predictionleaf.data(), try pypredictionleaf.data())
        XCTAssertEqual(try predictionleaf.dataShape(), try pypredictionleaf.dataShape())

        let predictionContributions = try booster.predict(from: test, predictionContributions: true)
        let pypredictionContributions = pyBooster.predict(data: pyTest, pred_contribs: true)
        XCTAssertEqual(try predictionContributions.data(), try pypredictionContributions.data())
        XCTAssertEqual(try predictionContributions.dataShape(), try pypredictionContributions.dataShape())

        let approximateContributions = try booster.predict(from: test, approximateContributions: true)
        let pyapproximateContributions = pyBooster.predict(data: pyTest, approx_contribs: true)
        XCTAssertEqual(try approximateContributions.data(), try pyapproximateContributions.data())
        XCTAssertEqual(try approximateContributions.dataShape(), try pyapproximateContributions.dataShape())

        let predictionInteractions = try booster.predict(from: test, predictionInteractions: true)
        let pypredictionInteractions = pyBooster.predict(data: pyTest, pred_interactions: true)
        XCTAssertEqual(try predictionInteractions.data(), try pypredictionInteractions.data())
        XCTAssertEqual(try predictionInteractions.dataShape(), try pypredictionInteractions.dataShape())
    }

    func testPredictOne() throws {
        let booster = try randomTrainedBooster()
        let pyBooster = try python(booster: booster)

        let features: [Float] = [1, 2, 3, 4, 5]
        let dmatrix = try DMatrix(name: "test", from: features, shape: Shape(1, 5))

        let pydmatrix = try python(dmatrix: dmatrix)

        let predicted = try booster.predict(features: features)
        let pypredicted = pyBooster.predict(pydmatrix)

        XCTAssertEqual(predicted, Float(pypredicted[0])!)
    }

    static var allTests = [
        ("testAttribute", testAttribute),
        ("testAttributes", testAttributes),
        ("testJsonDumped", testJsonDumped),
        ("testTextDumped", testTextDumped),
        ("testDotDumped", testDotDumped),
        ("testScoreEmptyFeatureMap", testScoreEmptyFeatureMap),
        ("testScoreWithFeatureMap", testScoreWithFeatureMap),
        ("testSystemLibraryVersion", testSystemLibraryVersion),
        ("testConfig", testConfig),
        ("testRawDumped", testRawDumped),
        ("testLoadModel", testLoadModel),
        ("testLoadConfig", testLoadConfig),
        ("testInitializeWithType", testInitializeWithType),
        ("testBoosterPredictOptionMasks", testBoosterPredictOptionMasks),
        ("testPredictOne", testPredictOne),
    ]
}
