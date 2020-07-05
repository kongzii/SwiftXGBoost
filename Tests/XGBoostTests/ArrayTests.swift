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

    static var allTests = [
        ("testThrowInvalidFeatureMap", testThrowInvalidFeatureMap),
    ]
}
