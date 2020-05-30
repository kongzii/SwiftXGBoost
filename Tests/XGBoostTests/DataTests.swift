import XCTest

@testable import XGBoost

final class DataTests: XCTestCase {
    func countsTest() throws {
        let randomArray = (0 ..< 10).map { _ in Float.random(in: 0 ..< 2) }
        let data = try Data(
            name: "data",
            values: randomArray,
            shape: Shape(5, 2),
            threads: 1
        )
        XCTAssertEqual(try data.rowCount(), 5)
        XCTAssertEqual(try data.columnCount(), 2)
        XCTAssertEqual(try data.shape().row, 5)
        XCTAssertEqual(try data.shape().column, 2)
    }

    static var allTests = [
        ("countsTest", countsTest),
    ]
}
