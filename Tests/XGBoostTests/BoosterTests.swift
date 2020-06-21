import PythonKit
import XCTest

@testable import XGBoost

private let PXGBOOST = Python.import("xgboost")
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

        let temporaryModelFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testJsonDumped.xgboost", isDirectory: false
        ).path
        let temporaryDumpFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testJsonDumped.json", isDirectory: false
        ).path

        try booster.save(to: temporaryModelFile)
        let pyBooster = PXGBOOST.Booster(model_file: temporaryModelFile)
        pyBooster.dump_model(fout: temporaryDumpFile, dump_format: "json")

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

        let temporaryModelFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testTextDumped.xgboost", isDirectory: false
        ).path
        let temporaryDumpFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testTextDumped.json", isDirectory: false
        ).path
        let temporaryFeatureMapFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testTextDumped.featureMap", isDirectory: false
        ).path

        let features = [Feature(name: "x", type: .quantitative)]
        try features.saveFeatureMap(to: temporaryFeatureMapFile)

        try booster.save(to: temporaryModelFile)
        let pyBooster = PXGBOOST.Booster(model_file: temporaryModelFile)
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

        let temporaryModelFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testDotDumped.xgboost", isDirectory: false
        ).path

        try booster.save(to: temporaryModelFile)
        let pyBooster = PXGBOOST.Booster(model_file: temporaryModelFile)

        let dot = try booster.rawDumped(format: .dot)
        let pyDot = pyBooster.get_dump(dump_format: "dot").map { String($0)! }

        XCTAssertEqual(dot, pyDot)
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

        let temporaryModelFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testScoreEmptyFeatureMap.xgboost", isDirectory: false
        ).absoluteString

        try booster.save(to: temporaryModelFile)
        let pyBooster = PXGBOOST.Booster(model_file: temporaryModelFile)

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

        let temporaryModelFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testScoreWithFeatureMap.xgboost", isDirectory: false
        ).path
        let temporaryNamesFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "testScoreWithFeatureMap.featuremap.txt", isDirectory: false
        ).path

        try features.saveFeatureMap(to: temporaryNamesFile)

        try booster.save(to: temporaryModelFile)
        let pyBooster = PXGBOOST.Booster(model_file: temporaryModelFile)

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

    static var allTests = [
        ("testAttribute", testAttribute),
        ("testAttributes", testAttributes),
        ("testJsonDumped", testJsonDumped),
        ("testTextDumped", testTextDumped),
        ("testDotDumped", testDotDumped),
        ("testScoreEmptyFeatureMap", testScoreEmptyFeatureMap),
        ("testScoreWithFeatureMap", testScoreWithFeatureMap),
    ]
}
