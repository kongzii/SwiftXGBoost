import PythonKit
import XCTest

@testable import XGBoost

let PXGBOOST = Python.import("xgboost")

func assertEqualDictionary(_ a: [String: Float], _ b: [String: Float], accuracy: Float) {
    for (key, value) in a {
        XCTAssertNotNil(b[key])
        XCTAssertEqual(value, b[key]!, accuracy: accuracy)
    }
}

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
        let jsonData = try booster.dumped(
            features: try data.features(),
            format: .json
        ).data(using: String.Encoding.utf8)!
        let jsonObject = try? JSONSerialization.jsonObject(with: jsonData)
        XCTAssertNotNil(jsonObject)
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
        let text = try booster.dumped(
            features: [Feature(name: "x", type: .quantitative)],
            format: .text
        )
        XCTAssertNotEqual(text, "")
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
        assertEqualDictionary(gainMap!, pyGainMap, accuracy: 1e-6)

        let pyTotalGainMap = [String: Float](pyBooster.get_score(
            fmap: "", importance_type: "total_gain"
        ))!
        let (_, totalGainMap) = try booster.score(featureMap: "", importance: .totalGain)
        assertEqualDictionary(totalGainMap!, pyTotalGainMap, accuracy: 1e-6)

        let pyCoverMap = [String: Float](pyBooster.get_score(
            fmap: "", importance_type: "cover"
        ))!
        let (_, coverMap) = try booster.score(featureMap: "", importance: .cover)
        assertEqualDictionary(coverMap!, pyCoverMap, accuracy: 1e-6)

        let pyTotalCoverMap = [String: Float](pyBooster.get_score(
            fmap: "", importance_type: "total_cover"
        ))!
        let (_, totalCoverMap) = try booster.score(featureMap: "", importance: .totalCover)
        assertEqualDictionary(totalCoverMap!, pyTotalCoverMap, accuracy: 1e-6)
    }

    func testScoreWithFeatureMap() throws {
        let randomArray = (0 ..< 50).map { _ in Float.random(in: 0 ..< 2) }
        let features = (0 ..< 5).map { Feature(name: "Feature-\($0)", type: .quantitative) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let data = try DMatrix(
            name: "data",
            from: randomArray,
            shape: Shape(10, 5),
            label: label,
            features: features,
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
        assertEqualDictionary(gainMap!, pyGainMap, accuracy: 1e-6)

        let pyTotalGainMap = [String: Float](pyBooster.get_score(
            fmap: temporaryNamesFile, importance_type: "total_gain"
        ))!
        let (_, totalGainMap) = try booster.score(
            featureMap: temporaryNamesFile, importance: .totalGain
        )
        assertEqualDictionary(totalGainMap!, pyTotalGainMap, accuracy: 1e-6)

        let pyCoverMap = [String: Float](pyBooster.get_score(
            fmap: temporaryNamesFile, importance_type: "cover"
        ))!
        let (_, coverMap) = try booster.score(
            featureMap: temporaryNamesFile, importance: .cover
        )
        assertEqualDictionary(coverMap!, pyCoverMap, accuracy: 1e-6)

        let pyTotalCoverMap = [String: Float](pyBooster.get_score(
            fmap: temporaryNamesFile, importance_type: "total_cover"
        ))!
        let (_, totalCoverMap) = try booster.score(
            featureMap: temporaryNamesFile, importance: .totalCover
        )
        assertEqualDictionary(totalCoverMap!, pyTotalCoverMap, accuracy: 1e-6)
    }

    static var allTests = [
        ("testAttribute", testAttribute),
        ("testAttributes", testAttributes),
        ("testJsonDumped", testJsonDumped),
        ("testTextDumped", testTextDumped),
        ("testScoreEmptyFeatureMap", testScoreEmptyFeatureMap),
        ("testScoreWithFeatureMap", testScoreWithFeatureMap),
    ]
}
