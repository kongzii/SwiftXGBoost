import PythonKit
import XCTest

func assertEqual(
    _ a: [String: Float],
    _ b: [String: Float],
    accuracy: Float
) {
    for (key, value) in a {
        XCTAssertNotNil(b[key])
        XCTAssertEqual(value, b[key]!, accuracy: accuracy)
    }
}

func assertEqual(
    _ a: [String: [Float]],
    _ b: [String: [Float]],
    accuracy: Float
) {
    for (key, values) in a {
        XCTAssertNotNil(b[key])
        for (valueA, valueB) in zip(values, b[key]!) {
            XCTAssertEqual(valueA, valueB, accuracy: accuracy)
        }
    }
}
