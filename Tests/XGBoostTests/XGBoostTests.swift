import XCTest

@testable import XGBoost

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
            features: [Feature(name: "x", type: .float)],
            format: .text
        )
        XCTAssertNotEqual(text, "")
    }

    static var allTests = [
        ("attributeTest", attributeTest),
        ("attributesTest", attributesTest),
        ("jsonDumpedTest", jsonDumpedTest),
        ("textDumpedTest", textDumpedTest),
    ]
}
