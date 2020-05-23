import XCTest

@testable import XGBoost

final class XGBoostTests: XCTestCase {
    func getAttributeTest() throws {
        let xgboost = try XGBoost()
        try xgboost.setAttribute(name: "testName", value: "testValue")
        XCTAssertEqual(try xgboost.getAttribute(name: "testName"), "testValue")
        XCTAssertEqual(try xgboost.getAttribute(name: "unknownName"), nil)
    }

    func getAttributesTest() throws {
        let xgboost = try XGBoost()
        try xgboost.setAttribute(name: "attribute1", value: "value1")
        try xgboost.setAttribute(name: "attribute2", value: "value2")
        XCTAssertEqual(try xgboost.getAttributes(), ["attribute1": "value1", "attribute2": "value2"])
    }

    static var allTests = [
        ("getAttributeTest", getAttributeTest),
        ("getAttributesTest", getAttributesTest),
    ]
}
