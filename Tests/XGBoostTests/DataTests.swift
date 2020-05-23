import XCTest

@testable import XGBoost

final class DataTests: XCTestCase {
    func countsTest() throws {
        let randomArray = (0 ..< 10).map { _ in Float.random(in: 0 ..< 2) }
        let data = try Data(
            name: "data",
            values: randomArray,
            rowCount: 5,
            columnCount: 2,
            threads: 1
        )
        XCTAssertEqual(try data.getRowCount(), 5)
        XCTAssertEqual(try data.getColumnCount(), 2)
    }

    static var allTests = [
        ("countsTest", countsTest),
    ]
}
