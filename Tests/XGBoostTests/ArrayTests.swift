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

    func testArrayWithShapeCount() {
        let array = ArrayWithShape([1, 2, 3, 4, 5, 6], shape: Shape(3, 2))
        XCTAssertEqual(array.count, 6)
    }

    func testArrayWithShapeShape() {
        let array = ArrayWithShape([1, 2, 3, 4, 5, 6], shape: Shape(3, 2))
        XCTAssertEqual(array.shape, Shape(3, 2))
    }

    func testArrayWithShapeSubscript() {
        var array = ArrayWithShape([1, 2, 3, 4, 5, 6], shape: Shape(3, 2))
        let arrayExpected = ArrayWithShape([2, 2, 3, 4, 5, 6], shape: Shape(3, 2))
        array[0] = 2
        XCTAssertEqual(array, arrayExpected)
    }

    func testArrayWithShapeDataComfortances() {
        let floatArray = ArrayWithShape<Float>([1, 2, 3], shape: Shape(3, 1))
        let intArray = ArrayWithShape<Int32>([1, 2, 3], shape: Shape(3, 1))
        let uintArray = ArrayWithShape<UInt32>([1, 2, 3], shape: Shape(3, 1))

        XCTAssertEqual(try floatArray.data(), [1, 2, 3])
        XCTAssertEqual(try intArray.data(), [1, 2, 3])
        XCTAssertEqual(try uintArray.data(), [1, 2, 3])
    }

    static var allTests = [
        ("testThrowInvalidFeatureMap", testThrowInvalidFeatureMap),
        ("testDiff", testDiff),
        ("testArrayWithShapeCount", testArrayWithShapeCount),
        ("testArrayWithShapeShape", testArrayWithShapeShape),
        ("testArrayWithShapeSubscript", testArrayWithShapeSubscript),
        ("testArrayWithShapeDataComfortances", testArrayWithShapeDataComfortances),
    ]
}
