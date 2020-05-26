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

final class XGBoostTests: XCTestCase {
    func attributeTest() throws {
        let xgboost = try XGBoost()
        try xgboost.set(attribute: "testName", value: "testValue")
        XCTAssertEqual(try xgboost.attribute(name: "testName"), "testValue")
        XCTAssertNil(try xgboost.attribute(name: "unknownName"))
    }

    func attributesTest() throws {
        let xgboost = try XGBoost()
        try xgboost.set(attribute: "attribute1", value: "value1")
        try xgboost.set(attribute: "attribute2", value: "value2")
        XCTAssertEqual(try xgboost.attributes(), ["attribute1": "value1", "attribute2": "value2"])
    }

    func jsonDumpedTest() throws {
        let randomArray = (0 ..< 10).map { _ in Float.random(in: 0 ..< 2) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let data = try Data(
            name: "data",
            values: randomArray,
            shape: (10, 1),
            label: label,
            threads: 1
        )
        let xgboost = try XGBoost(with: [data])
        try xgboost.train(
            iterations: 5,
            trainingData: data
        )
        let jsonData = try xgboost.dumped(
            features: try data.features(),
            format: .json
        ).data(using: String.Encoding.utf8)!
        let jsonObject = try? JSONSerialization.jsonObject(with: jsonData)
        XCTAssertNotNil(jsonObject)
    }

    func textDumpedTest() throws {
        let randomArray = (0 ..< 10).map { _ in Float.random(in: 0 ..< 2) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let data = try Data(
            name: "data",
            values: randomArray,
            shape: (10, 1),
            label: label,
            threads: 1
        )
        let xgboost = try XGBoost(with: [data])
        try xgboost.train(
            iterations: 5,
            trainingData: data
        )
        let text = try xgboost.dumped(
            features: [Feature(name: "x", type: .quantitative)],
            format: .text
        )
        XCTAssertNotEqual(text, "")
    }

    func scoreEmptyFeatureMapTest() throws {
        let randomArray = (0 ..< 50).map { _ in Float.random(in: 0 ..< 2) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let data = try Data(
            name: "data",
            values: randomArray,
            shape: (10, 5),
            label: label,
            threads: 1
        )
        let xgboost = try XGBoost(with: [data])
        try xgboost.train(
            iterations: 10,
            trainingData: data
        )

        let temporaryModelFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "scoreEmptyFeatureMapTest.xgboost", isDirectory: false
        ).absoluteString

        try xgboost.save(to: temporaryModelFile)
        let pyXgboost = PXGBOOST.Booster(model_file: temporaryModelFile)

        let pyWeightMap = [String: Int](pyXgboost.get_score(
            fmap: "", importance_type: "weight"
        ))!
        let (weightMap, _) = try xgboost.score(featureMap: "", importance: .weight)
        XCTAssertEqual(weightMap, pyWeightMap)

        let pyGainMap = [String: Float](pyXgboost.get_score(
            fmap: "", importance_type: "gain"
        ))!
        let (_, gainMap) = try xgboost.score(featureMap: "", importance: .gain)
        assertEqualDictionary(gainMap!, pyGainMap, accuracy: 1e-6)

        let pyTotalGainMap = [String: Float](pyXgboost.get_score(
            fmap: "", importance_type: "total_gain"
        ))!
        let (_, totalGainMap) = try xgboost.score(featureMap: "", importance: .totalGain)
        assertEqualDictionary(totalGainMap!, pyTotalGainMap, accuracy: 1e-6)

        let pyCoverMap = [String: Float](pyXgboost.get_score(
            fmap: "", importance_type: "cover"
        ))!
        let (_, coverMap) = try xgboost.score(featureMap: "", importance: .cover)
        assertEqualDictionary(coverMap!, pyCoverMap, accuracy: 1e-6)

        let pyTotalCoverMap = [String: Float](pyXgboost.get_score(
            fmap: "", importance_type: "total_cover"
        ))!
        let (_, totalCoverMap) = try xgboost.score(featureMap: "", importance: .totalCover)
        assertEqualDictionary(totalCoverMap!, pyTotalCoverMap, accuracy: 1e-6)
    }

    func scoreWithFeatureMapTest() throws {
        let randomArray = (0 ..< 50).map { _ in Float.random(in: 0 ..< 2) }
        let features = (0 ..< 5).map { Feature(name: "Feature-\($0)", type: .quantitative) }
        let label = (0 ..< 10).map { _ in Float([0, 1].randomElement()!) }
        let data = try Data(
            name: "data",
            values: randomArray,
            shape: (10, 5),
            label: label,
            features: features,
            threads: 1
        )
        let xgboost = try XGBoost(with: [data])
        try xgboost.train(
            iterations: 10,
            trainingData: data
        )

        let temporaryModelFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "scoreWithFeatureMapTest.xgboost", isDirectory: false
        ).path
        let temporaryNamesFile = FileManager.default.temporaryDirectory.appendingPathComponent(
            "scoreWithFeatureMapTest.featuremap.txt", isDirectory: false
        ).path

        try features.saveFeatureMap(to: temporaryNamesFile)

        try xgboost.save(to: temporaryModelFile)
        let pyXgboost = PXGBOOST.Booster(model_file: temporaryModelFile)

        let pyWeightMap = [String: Int](pyXgboost.get_score(
            fmap: temporaryNamesFile, importance_type: "weight"
        ))!
        let (weightMap, _) = try xgboost.score(
            featureMap: temporaryNamesFile, importance: .weight
        )
        XCTAssertEqual(weightMap, pyWeightMap)

        let pyGainMap = [String: Float](pyXgboost.get_score(
            fmap: temporaryNamesFile, importance_type: "gain"
        ))!
        let (_, gainMap) = try xgboost.score(
            featureMap: temporaryNamesFile, importance: .gain
        )
        assertEqualDictionary(gainMap!, pyGainMap, accuracy: 1e-6)

        let pyTotalGainMap = [String: Float](pyXgboost.get_score(
            fmap: temporaryNamesFile, importance_type: "total_gain"
        ))!
        let (_, totalGainMap) = try xgboost.score(
            featureMap: temporaryNamesFile, importance: .totalGain
        )
        assertEqualDictionary(totalGainMap!, pyTotalGainMap, accuracy: 1e-6)

        let pyCoverMap = [String: Float](pyXgboost.get_score(
            fmap: temporaryNamesFile, importance_type: "cover"
        ))!
        let (_, coverMap) = try xgboost.score(
            featureMap: temporaryNamesFile, importance: .cover
        )
        assertEqualDictionary(coverMap!, pyCoverMap, accuracy: 1e-6)

        let pyTotalCoverMap = [String: Float](pyXgboost.get_score(
            fmap: temporaryNamesFile, importance_type: "total_cover"
        ))!
        let (_, totalCoverMap) = try xgboost.score(
            featureMap: temporaryNamesFile, importance: .totalCover
        )
        assertEqualDictionary(totalCoverMap!, pyTotalCoverMap, accuracy: 1e-6)
    }

    static var allTests = [
        ("attributeTest", attributeTest),
        ("attributesTest", attributesTest),
        ("jsonDumpedTest", jsonDumpedTest),
        ("textDumpedTest", textDumpedTest),
        ("scoreEmptyFeatureMapTest", scoreEmptyFeatureMapTest),
        ("scoreWithFeatureMapTest", scoreWithFeatureMapTest),
    ]
}
