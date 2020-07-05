import XCTest

@testable import XGBoost

final class ArrayTests: XCTestCase {
    func testThrowInvalidFeatureMap() throws {
        let path = temporaryFile()
        let featureMap = """
        0 feature1 q
        1 feature2 u
        """

        try featureMap.write(
            toFile: path,
            atomically: true,
            encoding: .utf8
        )

        XCTAssertThrowsError(try [Feature](fromFeatureMap: path))
    }

    func testDiff() {
        let values = [1, 2, 3, 4, 10]
        let expectedDiff = [1, 1, 1, 6]
        XCTAssertEqual(values.diff(), expectedDiff)
    }

    static var allTests = [
        ("testThrowInvalidFeatureMap", testThrowInvalidFeatureMap),
        ("testDiff", testDiff),
    ]
}
