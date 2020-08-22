import XCTest

@testable import XGBoost

final class DMatrixTests: XCTestCase {
    func testSetNilFeatures() throws {
        var data = try randomDMatrix(columns: 1)
        try data.features()
        XCTAssertEqual(data._features, [Feature(name: "f0", type: .quantitative)])
        try data.set(features: nil)
        XCTAssertEqual(data._features, nil)
    }

    func testInvalidCacheFormat() throws {
        XCTAssertThrowsError(try DMatrix(
            name: "data",
            from: "file",
            format: .csv
        ))
    }

    func testInvalidDMatrixShape() throws {
        XCTAssertThrowsError(try DMatrix(
            name: "data",
            from: [1, 2, 3, 3, 4, 6],
            shape: Shape(6)
        ))
    }

    func testIsFlat() throws {
        let data = ArrayWithShape([10, 3, 1], shape: Shape(3))
        XCTAssertEqual(try data.isFlat(), true)
    }

    func testInitWithWeight() throws {
        let data = try DMatrix(
            name: "data",
            from: [1, 2, 3, 3, 4, 6],
            shape: Shape(3, 2),
            weight: [1, 2, 3]
        )
        XCTAssertEqual(try data.get(field: .weight), [1, 2, 3])
    }

    func testInitWithBaseMargin() throws {
        let data = try DMatrix(
            name: "data",
            from: [1, 2, 3, 3, 4, 6],
            shape: Shape(3, 2),
            baseMargin: [1, 2, 3]
        )
        XCTAssertEqual(try data.get(field: .baseMargin), [1, 2, 3])
    }

    func testInitWithBound() throws {
        let data = try DMatrix(
            name: "data",
            from: [1, 2, 3, 3, 4, 6],
            shape: Shape(3, 2),
            labelLowerBound: [1, 2, 3],
            labelUpperBound: [2, 3, 4]
        )
        XCTAssertEqual(try data.get(field: .labelLowerBound), [1, 2, 3])
        XCTAssertEqual(try data.get(field: .labelUpperBound), [2, 3, 4])
    }

    func testCounts() throws {
        let randomArray = (0 ..< 10).map { _ in Float.random(in: 0 ..< 2) }
        let data = try DMatrix(
            name: "data",
            from: randomArray,
            shape: Shape(5, 2),
            threads: 1
        )
        XCTAssertEqual(try data.rowCount(), 5)
        XCTAssertEqual(try data.columnCount(), 2)
        XCTAssertEqual(try data.shape(), [5, 2])
    }

    func testFromCSVFile() throws {
        let csvFile = temporaryFile()

        let fileContent = """
        1.0,2.0,3.0
        0.0,4.0,5.0
        1.0,6.0,7.0
        """

        try fileContent.write(
            toFile: csvFile,
            atomically: true,
            encoding: .utf8
        )

        let data = try DMatrix(
            name: "test",
            from: csvFile,
            format: .csv,
            labelColumn: 0
        )

        XCTAssertEqual(try data.get(field: .label), [1.0, 0.0, 1.0])
        XCTAssertEqual(try data.rowCount(), 3)
        XCTAssertEqual(try data.columnCount(), 2)
        XCTAssertEqual(try data.shape(), [3, 2])
    }

    func testSaveAndLoadBinary() throws {
        let data = try DMatrix(
            name: "data",
            from: [1, 2, 3, 3, 4, 6],
            shape: Shape(3, 2),
            label: [1, 0, 1]
        )

        let path = temporaryFile()

        try data.save(to: path)

        let loadedData = try DMatrix(
            name: "loaded",
            from: path,
            format: .binary
        )

        XCTAssertEqual(try data.get(field: .label), try loadedData.get(field: .label))
        XCTAssertEqual(try data.rowCount(), try loadedData.rowCount())
        XCTAssertEqual(try data.columnCount(), try loadedData.columnCount())
        XCTAssertEqual(try data.shape(), try loadedData.shape())
        XCTAssertEqual(try data.shape(), [3, 2])
    }

    func testSaveAndLoadFeatureMap() throws {
        let features = [
            Feature("x", .quantitative),
            Feature("y", .quantitative),
        ]

        let path = temporaryFile()

        try features.saveFeatureMap(to: path)

        let loadedFeatures = try [Feature](fromFeatureMap: path)

        XCTAssertEqual(features, loadedFeatures)

        let data = try DMatrix(
            name: "data",
            from: [1, 2, 3, 3, 4, 6],
            shape: Shape(3, 2),
            label: [1, 0, 1]
        )
        try data.loadFeatureMap(from: path)

        XCTAssertEqual(try data.features(), loadedFeatures)
    }

    func testSlice() throws {
        let data = try DMatrix(
            name: "data",
            from: [1, 2, 3, 3, 4, 6],
            shape: Shape(3, 2),
            label: [1, 0, 1]
        )

        let firstSlice = try data.slice(
            indexes: [0, 2],
            newName: "firstSlice"
        )
        XCTAssertEqual(firstSlice.name, "firstSlice")
        XCTAssertEqual(try firstSlice.shape(), [2, 2])
        XCTAssertEqual(try firstSlice.get(field: .label), [1, 1])

        let secondSlice = try data.slice(
            indexes: [1],
            newName: "secondSlice"
        )
        XCTAssertEqual(secondSlice.name, "secondSlice")
        XCTAssertEqual(try secondSlice.shape(), [1, 2])
        XCTAssertEqual(try secondSlice.get(field: .label), [0])

        let thirdSlice = try data.slice(
            indexes: 0 ..< 2,
            newName: "thirdSlice"
        )
        XCTAssertEqual(thirdSlice.name, "thirdSlice")
        XCTAssertEqual(try thirdSlice.shape(), [2, 2])
        XCTAssertEqual(try thirdSlice.get(field: .label), [1, 0])
    }

    func testSetGetFloatInfo() throws {
        let data = try DMatrix(
            name: "data",
            from: [1, 2, 3, 3, 4, 6],
            shape: Shape(3, 2)
        )

        try data.set(field: .label, values: [1, 2, 3])
        XCTAssertEqual(try data.get(field: .label), [1, 2, 3])

        try data.set(field: .weight, values: [1, 3, 3])
        XCTAssertEqual(try data.get(field: .weight), [1, 3, 3])
    }

    func testSetGetUIntInfo() throws {
        let data = try DMatrix(
            name: "data",
            from: [1, 2, 3, 3, 4, 6],
            shape: Shape(6, 1)
        )

        try data.set(field: .group, values: [1, 1, 1, 3])
        XCTAssertEqual(try data.get(field: .groupPtr), [0, 1, 2, 3, 6])
    }

    static var allTests = [
        ("testCounts", testCounts),
        ("testFromCSVFile", testFromCSVFile),
        ("testSaveAndLoadBinary", testSaveAndLoadBinary),
        ("testSaveAndLoadFeatureMap", testSaveAndLoadFeatureMap),
        ("testSlice", testSlice),
        ("testSetGetFloatInfo", testSetGetFloatInfo),
        ("testSetGetUIntInfo", testSetGetUIntInfo),
    ]
}
