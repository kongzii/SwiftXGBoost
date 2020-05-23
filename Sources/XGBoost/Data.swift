import CXGBoost

public class Data {
    public var name: String
    public var featuresNames: [String]?
    public var dmatrix: UnsafeMutablePointer<DMatrixHandle?>

    public var pointee: DMatrixHandle? {
        dmatrix.pointee
    }

    public init(
        name: String,
        featuresNames: [String]? = nil,
        dmatrix: UnsafeMutablePointer<DMatrixHandle?>
    ) {
        self.name = name
        self.featuresNames = featuresNames
        self.dmatrix = dmatrix
    }

    public init(
        name: String,
        values: [Float],
        rowCount: Int,
        columnCount: Int,
        featuresNames: [String]? = nil,
        missingValue: Float = Float.greatestFiniteMagnitude,
        threads: Int = 0
    ) throws {
        self.name = name
        self.featuresNames = featuresNames
        dmatrix = .allocate(capacity: 1)

        try safe {
            XGDMatrixCreateFromMat_omp(
                values,
                UInt64(rowCount),
                UInt64(columnCount),
                missingValue,
                dmatrix,
                Int32(threads)
            )
        }
    }

    public init(
        name: String,
        file: String,
        format: DataFormat = .csv,
        featuresNames: [String]? = nil,
        labelColumn: Int? = nil,
        silent: Bool = true,
        fileQuery: [String] = []
    ) throws {
        self.name = name
        self.featuresNames = featuresNames
        dmatrix = .allocate(capacity: 1)

        var fileQuery = fileQuery

        if format == .csv {
            fileQuery.append("format=\(format)")
        }

        if let labelColumn = labelColumn {
            fileQuery.append("label_column=\(labelColumn)")
        }

        try safe {
            XGDMatrixCreateFromFile(
                file + (fileQuery.isEmpty ? "" : "?\(fileQuery.joined(separator: "&"))"),
                silent ? 1 : 0,
                dmatrix
            )
        }
    }

    deinit {
        try! safe {
            XGDMatrixFree(pointee)
        }
    }

    public func getRowCount() throws -> Int {
        var count: UInt64 = 0
        try! safe {
            XGDMatrixNumRow(
                pointee,
                &count
            )
        }
        return Int(count)
    }

    public func getColumnCount() throws -> Int {
        var count: UInt64 = 0
        try! safe {
            XGDMatrixNumCol(
                pointee,
                &count
            )
        }
        return Int(count)
    }

    public func save(
        to path: String,
        silent: Bool = true
    ) throws {
        try safe {
            XGDMatrixSaveBinary(
                pointee,
                path,
                silent ? 1 : 0
            )
        }
    }

    public func slice(
        indexes: [Int],
        newName: String? = nil,
        allowGroups: Bool = false
    ) throws -> Data {
        let indexes: [Int32] = indexes.map { Int32($0) }
        let slicedDmatrix = UnsafeMutablePointer<DMatrixHandle?>.allocate(capacity: 1)

        try safe {
            XGDMatrixSliceDMatrixEx(
                pointee,
                indexes,
                UInt64(indexes.count),
                slicedDmatrix,
                allowGroups ? 1 : 0
            )
        }

        return Data(
            name: newName ?? name,
            featuresNames: featuresNames,
            dmatrix: slicedDmatrix
        )
    }

    public func slice(
        indexes: Range<Int>,
        newName: String? = nil,
        allowGroups: Bool = false
    ) throws -> Data {
        try slice(
            indexes: Array(indexes),
            newName: newName,
            allowGroups: allowGroups
        )
    }

    public func setUIntInfo(
        field: Field,
        values: [UInt32]
    ) throws {
        try safe {
            XGDMatrixSetUIntInfo(
                pointee,
                field.rawValue,
                values,
                UInt64(values.count)
            )
        }
    }

    public func getUIntInfo(
        field: Field
    ) throws -> [Int] {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafePointer<UInt32>?>.allocate(capacity: 1)

        try safe {
            XGDMatrixGetUIntInfo(pointee, field.rawValue, outLenght, outResult)
        }

        return (0 ..< Int(outLenght.pointee)).lazy.map { Int(outResult.pointee![$0]) }
    }

    public func setFloatInfo(
        field: Field,
        values: [Float]
    ) throws {
        try safe {
            XGDMatrixSetFloatInfo(
                pointee,
                field.rawValue,
                values,
                UInt64(values.count)
            )
        }
    }

    public func getFloatInfo(
        field: Field
    ) throws -> [Float] {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafePointer<Float>?>.allocate(capacity: 1)

        try safe {
            XGDMatrixGetFloatInfo(pointee, field.rawValue, outLenght, outResult)
        }

        return (0 ..< Int(outLenght.pointee)).lazy.map { outResult.pointee![$0] }
    }
}
