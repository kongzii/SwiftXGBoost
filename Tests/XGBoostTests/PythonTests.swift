import XCTest

@testable import XGBoost

final class PythonTests: XCTestCase {
    func testMakeNumpyArrayWithShape() throws {
        let values = [1.0, 2.0, 3.0, 4.0]
        let shape = Shape(2, 2)
        let npValues = values.makeNumpyArray(shape: shape)
        XCTAssertEqual(try npValues.dataShape(), shape)
    }

    static var allTests = [
        ("testMakeNumpyArrayWithShape", testMakeNumpyArrayWithShape),
    ]
}
