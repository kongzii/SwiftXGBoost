import CXGBoost

/// Data class used with XGBoost.
///
/// Data is encapsulation of DMatrixHandle, internal structure used by XGBoost,
/// which is optimized for both memory efficiency and training speed.
public class Data {
    /// Name of dataset, for example, "train".
    public var name: String

    var _features: [Feature]?
    var dmatrix: UnsafeMutablePointer<DMatrixHandle?>

    /// Pointer to underlying DMatrixHandle.
    public var pointee: DMatrixHandle? {
        dmatrix.pointee
    }

    /// Initialize Data with an existing DMatrixHandle pointer.
    ///
    /// - Parameter name: Name of dataset.
    /// - Parameter features: Array describing features in this dataset.
    /// - Parameter dmatrix: DMatrixHandle pointer.
    public init(
        name: String,
        features: [Feature]? = nil,
        dmatrix: UnsafeMutablePointer<DMatrixHandle?>
    ) throws {
        self.name = name
        self.dmatrix = dmatrix

        if let features = features {
            try set(features: features)
        }
    }

    /// Initialize Data.
    ///
    /// - Parameter name: Name of dataset.
    /// - Parameter values: Values source.
    /// - Parameter shape: Shape of resulting DMatrixHandle.
    /// - Parameter weight: Weight for each instance.
    /// - Parameter baseMargin: Set base margin of booster to start from.
    /// - Parameter features: Names and types of features.
    /// - Parameter missingValue: Value in the input data which needs to be present as a missing value.
    /// - Parameter threads:  Number of threads to use for loading data when parallelization is applicable. If 0, uses maximum threads available on the system.
    public init(
        name: String,
        values: [Float],
        shape: Shape,
        label: [Float]? = nil,
        weight: [Float]? = nil,
        baseMargin: [Float]? = nil,
        features: [Feature]? = nil,
        missingValue: Float = Float.greatestFiniteMagnitude,
        threads: Int = 0
    ) throws {
        self.name = name
        dmatrix = .allocate(capacity: 1)

        try safe {
            XGDMatrixCreateFromMat_omp(
                values,
                UInt64(shape.row),
                UInt64(shape.column),
                missingValue,
                dmatrix,
                Int32(threads)
            )
        }

        if let label = label {
            try set(label: label)
        }

        if let weight = weight {
            try set(weight: weight)
        }

        if let baseMargin = baseMargin {
            try set(baseMargin: baseMargin)
        }

        if let features = features {
            try set(features: features)
        }
    }

    /// Initialize Data from file.
    ///
    /// - Parameter name: Name of dataset.
    /// - Parameter file: File to laod from.
    /// - Parameter format: Format of input file.
    /// - Parameter features: Names and types of features.
    /// - Parameter labelColumn: Which column is for label.
    /// - Parameter label: Array of labels for data.
    /// - Parameter weight: Weight for each instance.
    /// - Parameter baseMargin: Set base margin of booster to start from.
    /// - Parameter silent: Whether print messages during construction.
    /// - Parameter fileQuery: Additional parameters that will be appended to the file path as query.
    public init(
        name: String,
        file: String,
        format: DataFormat = .csv,
        features: [Feature]? = nil,
        labelColumn: Int? = nil,
        label: [Float]? = nil,
        weight: [Float]? = nil,
        baseMargin: [Float]? = nil,
        silent: Bool = true,
        fileQuery: [String] = []
    ) throws {
        self.name = name
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

        if let label = label {
            try set(label: label)
        }

        if let weight = weight {
            try set(weight: weight)
        }

        if let baseMargin = baseMargin {
            try set(baseMargin: baseMargin)
        }

        if let features = features {
            try set(features: features)
        }
    }

    deinit {
        try! safe {
            XGDMatrixFree(pointee)
        }
    }

    /// Save DMatrixHandle to binary file.
    ///
    /// - Parameter to: File path.
    /// - Parameter silent: Whether print messages during construction.
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

    /// Save feature map compatible with XGBoost`s inputs.
    ///
    /// - Parameter to: Path where feature map will be saved.
    public func saveFeatureMap(
        to path: String
    ) throws {
        try features().saveFeatureMap(to: path)
    }

    /// - Returns: The number of rows.
    public func rowCount() throws -> Int {
        var count: UInt64 = 0
        try! safe {
            XGDMatrixNumRow(
                pointee,
                &count
            )
        }
        return Int(count)
    }

    /// - Returns: The number of columns.
    public func columnCount() throws -> Int {
        var count: UInt64 = 0
        try! safe {
            XGDMatrixNumCol(
                pointee,
                &count
            )
        }
        return Int(count)
    }

    /// - Returns: The shape of data, i.e. (rowCount(), columnCount()).
    public func shape() throws -> Shape {
        Shape(
            row: try rowCount(),
            column: try columnCount()
        )
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

        return try Data(
            name: newName ?? name,
            features: _features,
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
        field: String,
        values: [UInt32]
    ) throws {
        try safe {
            XGDMatrixSetUIntInfo(
                pointee,
                field,
                values,
                UInt64(values.count)
            )
        }
    }

    public func setFloatInfo(
        field: String,
        values: [Float]
    ) throws {
        try safe {
            XGDMatrixSetFloatInfo(
                pointee,
                field,
                values,
                UInt64(values.count)
            )
        }
    }

    public func getUIntInfo(
        field: UIntField
    ) throws -> [UInt32] {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafePointer<UInt32>?>.allocate(capacity: 1)

        try safe {
            XGDMatrixGetUIntInfo(pointee, field.rawValue, outLenght, outResult)
        }

        return (0 ..< Int(outLenght.pointee)).lazy.map { outResult.pointee![$0] }
    }

    public func getFloatInfo(
        field: FloatField
    ) throws -> [Float] {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafePointer<Float>?>.allocate(capacity: 1)

        try safe {
            XGDMatrixGetFloatInfo(pointee, field.rawValue, outLenght, outResult)
        }

        return (0 ..< Int(outLenght.pointee)).lazy.map { outResult.pointee![$0] }
    }

    public func set(
        field: UIntField,
        values: [UInt32]
    ) throws {
        try setUIntInfo(field: field.rawValue, values: values)
    }

    public func set(
        field: FloatField,
        values: [Float]
    ) throws {
        try setFloatInfo(field: field.rawValue, values: values)
    }

    public func set(
        label: [Float]
    ) throws {
        try set(field: .label, values: label)
    }

    public func set(
        weight: [Float]
    ) throws {
        try set(field: .weight, values: weight)
    }

    public func set(
        baseMargin: [Float]
    ) throws {
        try set(field: .baseMargin, values: baseMargin)
    }

    public func set(
        group: [UInt32]
    ) throws {
        try set(field: .group, values: group)
    }

    public func set(
        features: [Feature]?
    ) throws {
        guard let features = features else {
            _features = nil
            return
        }

        let columnCount = try self.columnCount()

        if features.count != columnCount {
            throw ValueError.runtimeError("Features count \(features.count) != data count \(columnCount).")
        }

        let names = features.map { $0.name }

        if names.count != Set(names).count {
            throw ValueError.runtimeError("Feature names must be unique.")
        }

        if !names.allSatisfy({ !$0.contains("[") && !$0.contains("]") && !$0.contains("<") }) {
            throw ValueError.runtimeError("Feature names must not contain [, ] or <.")
        }

        _features = features
    }

    public func label() throws -> [Float] {
        try getFloatInfo(field: .label)
    }

    public func weight() throws -> [Float] {
        try getFloatInfo(field: .weight)
    }

    public func baseMargin() throws -> [Float] {
        try getFloatInfo(field: .baseMargin)
    }

    public func group() throws -> [UInt32] {
        try getUIntInfo(field: .group)
    }

    public func features() throws -> [Feature] {
        if let features = _features {
            return features
        }

        return try (0 ..< columnCount()).map { Feature(name: String($0), type: .quantitative) }
    }
}
