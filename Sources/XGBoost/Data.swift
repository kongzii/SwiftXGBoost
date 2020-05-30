import CXGBoost

/// Convient alias for Python data name.
public typealias DMatrix = Data

/// Data class used with XGBoost.
///
/// Data is encapsulation of DMatrixHandle, internal structure used by XGBoost,
/// which is optimized for both memory efficiency and training speed.
public class Data {
    /// Name of dataset, for example, "train".
    public var name: String

    var _features: [Feature]?

    /// Pointer to underlying DMatrixHandle.
    public let dmatrix: DMatrixHandle?

    /// Initialize Data with an existing DMatrixHandle pointer.
    ///
    /// - Parameter name: Name of dataset.
    /// - Parameter features: Array describing features in this dataset.
    /// - Parameter dmatrix: DMatrixHandle pointer.
    public init(
        name: String,
        dmatrix: DMatrixHandle?,
        features: [Feature]? = nil,
        label: [Float]? = nil,
        weight: [Float]? = nil,
        baseMargin: [Float]? = nil
    ) throws {
        self.name = name
        self.dmatrix = dmatrix

        if let features = features {
            try set(features: features)
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
    }

    /// Initialize Data.
    ///
    /// - Parameter name: Name of dataset.
    /// - Parameter values: Values source.
    /// - Parameter shape: Shape of resulting DMatrixHandle.
    /// - Parameter label: Array of labels for data.
    /// - Parameter weight: Weight for each instance.
    /// - Parameter baseMargin: Set base margin of booster to start from.
    /// - Parameter features: Names and types of features.
    /// - Parameter missingValue: Value in the input data which needs to be present as a missing value.
    /// - Parameter threads:  Number of threads to use for loading data when parallelization is applicable. If 0, uses maximum threads available on the system.
    public convenience init(
        name: String,
        values: UnsafePointer<Float>,
        shape: Shape,
        label: [Float]? = nil,
        weight: [Float]? = nil,
        baseMargin: [Float]? = nil,
        features: [Feature]? = nil,
        missingValue: Float = Float.greatestFiniteMagnitude,
        threads: Int = 0
    ) throws {
        var dmatrix: DMatrixHandle?
        try safe {
            XGDMatrixCreateFromMat_omp(
                values,
                UInt64(shape.row),
                UInt64(shape.column),
                missingValue,
                &dmatrix,
                Int32(threads)
            )
        }

        try self.init(
            name: name,
            dmatrix: dmatrix,
            features: features,
            label: label,
            weight: weight,
            baseMargin: baseMargin
        )
    }

    /// Initialize Data.
    ///
    /// - Parameter name: Name of dataset.
    /// - Parameter values: Values source.
    /// - Parameter shape: Shape of resulting DMatrixHandle.
    /// - Parameter label: Array of labels for data.
    /// - Parameter weight: Weight for each instance.
    /// - Parameter baseMargin: Set base margin of booster to start from.
    /// - Parameter features: Names and types of features.
    /// - Parameter missingValue: Value in the input data which needs to be present as a missing value.
    /// - Parameter threads:  Number of threads to use for loading data when parallelization is applicable. If 0, uses maximum threads available on the system.
    public convenience init(
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
        var values = values
        try self.init(
            name: name,
            values: &values,
            shape: shape,
            label: label,
            weight: weight,
            baseMargin: baseMargin,
            features: features,
            missingValue: missingValue,
            threads: threads
        )
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
    public convenience init(
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
        var fileQuery = fileQuery

        if format == .csv {
            fileQuery.append("format=\(format)")
        }

        if let labelColumn = labelColumn {
            fileQuery.append("label_column=\(labelColumn)")
        }

        var dmatrix: DMatrixHandle?
        try safe {
            XGDMatrixCreateFromFile(
                file + (fileQuery.isEmpty ? "" : "?\(fileQuery.joined(separator: "&"))"),
                silent ? 1 : 0,
                &dmatrix
            )
        }

        try self.init(
            name: name,
            dmatrix: dmatrix,
            features: features,
            label: label,
            weight: weight,
            baseMargin: baseMargin
        )
    }

    deinit {
        try! safe {
            XGDMatrixFree(dmatrix)
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
                dmatrix,
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
                dmatrix,
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
                dmatrix,
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

    /// Slice and return a new Data that only contains `indexes`.
    ///
    /// - Parameter indexes: Array of indices to be selected.
    /// - Parameter newName: New name of the returned Data.
    /// - Parameter allowGroups: Allow slicing of a matrix with a groups attribute.
    /// - Returns: A new data class containing only selected indexes.
    public func slice(
        indexes: [Int],
        newName: String? = nil,
        allowGroups: Bool = false
    ) throws -> Data {
        let indexes: [Int32] = indexes.map { Int32($0) }
        var slicedDmatrix: DMatrixHandle?

        try safe {
            XGDMatrixSliceDMatrixEx(
                dmatrix,
                indexes,
                UInt64(indexes.count),
                &slicedDmatrix,
                allowGroups ? 1 : 0
            )
        }

        return try Data(
            name: newName ?? name,
            dmatrix: slicedDmatrix,
            features: _features
        )
    }

    /// Slice and return a new Data that only indexes from given range..
    ///
    /// - Parameter indexes: Range of indices to be selected.
    /// - Parameter newName: New name of the returned Data.
    /// - Parameter allowGroups: Allow slicing of a matrix with a groups attribute.
    /// - Returns: A new data class containing only selected indexes.
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

    /// Set uint type property into the DMatrixHandle.
    ///
    /// - Parameter field: The field name of the information.
    /// - Parameter values: The array of data to be set.
    public func setUIntInfo(
        field: String,
        values: [UInt32]
    ) throws {
        try safe {
            XGDMatrixSetUIntInfo(
                dmatrix,
                field,
                values,
                UInt64(values.count)
            )
        }
    }

    /// Set float type property into the DMatrixHandle.
    ///
    /// - Parameter field: The field name of the information.
    /// - Parameter values: The array of data to be set.
    public func setFloatInfo(
        field: String,
        values: [Float]
    ) throws {
        try safe {
            XGDMatrixSetFloatInfo(
                dmatrix,
                field,
                values,
                UInt64(values.count)
            )
        }
    }

    /// Get unsigned integer property from the DMatrixHandle.
    ///
    /// - Parameter field: The field name of the information.
    /// - Returns: An array of unsigned integer information of the data.
    public func getUIntInfo(
        field: UIntField
    ) throws -> [UInt32] {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafePointer<UInt32>?>.allocate(capacity: 1)

        try safe {
            XGDMatrixGetUIntInfo(dmatrix, field.rawValue, outLenght, outResult)
        }

        return (0 ..< Int(outLenght.pointee)).lazy.map { outResult.pointee![$0] }
    }

    /// Get float property from the DMatrixHandle.
    ///
    /// - Parameter field: The field name of the information.
    /// - Returns: An array of float information of the data.
    public func getFloatInfo(
        field: FloatField
    ) throws -> [Float] {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafePointer<Float>?>.allocate(capacity: 1)

        try safe {
            XGDMatrixGetFloatInfo(dmatrix, field.rawValue, outLenght, outResult)
        }

        return (0 ..< Int(outLenght.pointee)).lazy.map { outResult.pointee![$0] }
    }

    /// Set uint type property into the DMatrixHandle.
    ///
    /// - Parameter field: The field name of the information.
    /// - Parameter values: The array of data to be set.
    public func set(
        field: UIntField,
        values: [UInt32]
    ) throws {
        try setUIntInfo(field: field.rawValue, values: values)
    }

    /// Set float type property into the DMatrixHandle.
    ///
    /// - Parameter field: The field name of the information.
    /// - Parameter values: The array of data to be set.
    public func set(
        field: FloatField,
        values: [Float]
    ) throws {
        try setFloatInfo(field: field.rawValue, values: values)
    }

    /// Set float type property named "label" into the DMatrixHandle.
    ///
    /// - Parameter label: The array of labels to be set.
    public func set(
        label: [Float]
    ) throws {
        try set(field: .label, values: label)
    }

    /// Set float type property named "weight" into the DMatrixHandle.
    ///
    /// - Parameter label: The array of weights to be set.
    public func set(
        weight: [Float]
    ) throws {
        try set(field: .weight, values: weight)
    }

    /// Set float type property named "base_margin" into the DMatrixHandle.
    ///
    /// - Parameter label: The array of values to be set.
    public func set(
        baseMargin: [Float]
    ) throws {
        try set(field: .baseMargin, values: baseMargin)
    }

    /// Set uint type property named "group" into the DMatrixHandle.
    ///
    /// - Parameter label: The array of values to be set.
    public func set(
        group: [UInt32]
    ) throws {
        try set(field: .group, values: group)
    }

    /// Save names and types of features.
    ///
    /// - Parameter features: Optional array of features.
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

    /// - Returns: An array of label information of the data.
    public func label() throws -> [Float] {
        try getFloatInfo(field: .label)
    }

    /// - Returns: An array of label information of the data.
    public func labelUpperBound() throws -> [Float] {
        try getFloatInfo(field: .labelUpperBound)
    }

    /// - Returns: An array of label information of the data.
    public func labelLowerBound() throws -> [Float] {
        try getFloatInfo(field: .labelLowerBound)
    }

    /// - Returns: An array of weight information of the data.
    public func weight() throws -> [Float] {
        try getFloatInfo(field: .weight)
    }

    /// - Returns: An array of base margin information of the data.
    public func baseMargin() throws -> [Float] {
        try getFloatInfo(field: .baseMargin)
    }

    /// - Returns: An array of group information of the data.
    public func group() throws -> [UInt32] {
        try getUIntInfo(field: .group)
    }

    /// - Returns: Features of data, generates universal names if not previously set by user.
    public func features() throws -> [Feature] {
        if let features = _features {
            return features
        }

        return try (0 ..< columnCount()).map { Feature(name: String($0), type: .quantitative) }
    }
}
