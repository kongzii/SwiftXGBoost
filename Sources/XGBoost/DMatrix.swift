import CXGBoost

/// Backward compatible alias for Data
public typealias Data = DMatrix

/// Protocol used where [Float] values are expected.
public protocol FloatData {
    /// - Returns: Value as [Float]
    func data() throws -> [Float]
}

/// Protocol used where [Int32] values are expected.
public protocol Int32Data {
    /// - Returns: Value as [Int32]
    func data() throws -> [Int32]
}

/// Protocol used where [UInt32] values are expected.
public protocol UInt32Data {
    /// - Returns: Value as [UInt32]
    func data() throws -> [UInt32]
}

/// Protocol declaring that structure has shape.
public protocol ShapeData {
    /// - Returns: Shape of self.
    func dataShape() throws -> Shape
}

extension ShapeData {
    /// - Returns: Whether shape indicates flat structure.
    public func isFlat() throws -> Bool {
        try dataShape().row == 1
    }
}

/// Data class used with Booster.
///
/// Data is encapsulation of DMatrixHandle, internal structure used by XGBoost,
/// which is optimized for both memory efficiency and training speed.
public class DMatrix {
    /// Name of dataset, for example, "train".
    public var name: String

    var _features: [Feature]?

    /// Pointer to underlying DMatrixHandle.
    public let dmatrix: DMatrixHandle?

    /// Initialize Data with an existing DMatrixHandle pointer.
    ///
    /// - Parameter name: Name of dataset.
    /// - Parameter dmatrix: DMatrixHandle pointer.
    /// - Parameter features: Array describing features in this dataset.
    /// - Parameter label: Sets label after DMatrix initialization.
    /// - Parameter weight: Sets weight after DMatrix initialization.
    /// - Parameter baseMargin: Sets baseMargin after DMatrix initialization.
    /// - Parameter labelLowerBound: Sets labelLowerBound after DMatrix initialization.
    /// - Parameter labelUpperBound: Sets labelUpperBound after DMatrix initialization.
    public init(
        name: String,
        dmatrix: DMatrixHandle?,
        features: [Feature]?,
        label: FloatData?,
        weight: FloatData?,
        baseMargin: FloatData?,
        labelLowerBound: FloatData?,
        labelUpperBound: FloatData?
    ) throws {
        self.name = name
        self.dmatrix = dmatrix

        if let features = features {
            try set(features: features)
        }

        if let label = label {
            try set(field: .label, values: label)
        }

        if let weight = weight {
            try set(field: .weight, values: weight)
        }

        if let baseMargin = baseMargin {
            try set(field: .baseMargin, values: baseMargin)
        }

        if let labelLowerBound = labelLowerBound {
            try set(field: .labelLowerBound, values: labelLowerBound)
        }

        if let labelUpperBound = labelUpperBound {
            try set(field: .labelUpperBound, values: labelUpperBound)
        }
    }

    /// Initialize Data.
    ///
    /// - Parameter name: Name of dataset.
    /// - Parameter from: Values source.
    /// - Parameter shape: Shape of resulting DMatrixHandle.
    /// - Parameter features: Names and types of features.
    /// - Parameter label: Sets label after DMatrix initialization.
    /// - Parameter weight: Sets weight after DMatrix initialization.
    /// - Parameter baseMargin: Sets baseMargin after DMatrix initialization.
    /// - Parameter labelLowerBound: Sets labelLowerBound after DMatrix initialization.
    /// - Parameter labelUpperBound: Sets labelUpperBound after DMatrix initialization.
    /// - Parameter missingValue: Value in the input data which needs to be present as a missing value.
    /// - Parameter threads:  Number of threads to use for loading data when parallelization is applicable. If 0, uses maximum threads available on the system.
    public convenience init(
        name: String,
        from data: FloatData,
        shape: Shape,
        features: [Feature]? = nil,
        label: FloatData? = nil,
        weight: FloatData? = nil,
        baseMargin: FloatData? = nil,
        labelLowerBound: FloatData? = nil,
        labelUpperBound: FloatData? = nil,
        missingValue: Float = Float.greatestFiniteMagnitude,
        threads: Int = 0
    ) throws {
        var dmatrix: DMatrixHandle?
        try safe {
            XGDMatrixCreateFromMat_omp(
                try data.data(),
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
            baseMargin: baseMargin,
            labelLowerBound: labelLowerBound,
            labelUpperBound: labelUpperBound
        )
    }

    /// Initialize Data.
    ///
    /// - Parameter name: Name of dataset.
    /// - Parameter from: Data with shape comfortance.
    /// - Parameter features: Names and types of features.
    /// - Parameter label: Sets label after DMatrix initialization.
    /// - Parameter weight: Sets weight after DMatrix initialization.
    /// - Parameter baseMargin: Sets baseMargin after DMatrix initialization.
    /// - Parameter labelLowerBound: Sets labelLowerBound after DMatrix initialization.
    /// - Parameter labelUpperBound: Sets labelUpperBound after DMatrix initialization.
    /// - Parameter missingValue: Value in the input data which needs to be present as a missing value.
    /// - Parameter threads:  Number of threads to use for loading data when parallelization is applicable. If 0, uses maximum threads available on the system.
    public convenience init(
        name: String,
        from: FloatData & ShapeData,
        features: [Feature]? = nil,
        label: FloatData? = nil,
        weight: FloatData? = nil,
        baseMargin: FloatData? = nil,
        labelLowerBound: FloatData? = nil,
        labelUpperBound: FloatData? = nil,
        missingValue _: Float = Float.greatestFiniteMagnitude,
        threads: Int = 0
    ) throws {
        try self.init(
            name: name,
            from: from,
            shape: try from.dataShape(),
            features: features,
            label: label,
            weight: weight,
            baseMargin: baseMargin,
            labelLowerBound: labelLowerBound,
            labelUpperBound: labelUpperBound,
            threads: threads
        )
    }

    /// Initialize Data from file.
    ///
    /// - Parameter name: Name of dataset.
    /// - Parameter from: File to laod from.
    /// - Parameter format: Format of input file.
    /// - Parameter useCache: Use external memory.
    /// - Parameter features: Names and types of features.
    /// - Parameter labelColumn: Which column is for label.
    /// - Parameter label: Sets label after DMatrix initialization.
    /// - Parameter weight: Sets weight after DMatrix initialization.
    /// - Parameter baseMargin: Sets baseMargin after DMatrix initialization.
    /// - Parameter labelLowerBound: Sets labelLowerBound after DMatrix initialization.
    /// - Parameter labelUpperBound: Sets labelUpperBound after DMatrix initialization.
    /// - Parameter silent: Whether print messages during construction.
    /// - Parameter fileQuery: Additional parameters that will be appended to the file path as query.
    public convenience init(
        name: String,
        from file: String,
        format: DataFormat,
        useCache: Bool = false,
        features: [Feature]? = nil,
        labelColumn: Int? = nil,
        label: FloatData? = nil,
        weight: FloatData? = nil,
        baseMargin: FloatData? = nil,
        labelLowerBound: FloatData? = nil,
        labelUpperBound: FloatData? = nil,
        silent: Bool = true,
        fileQuery: [String] = []
    ) throws {
        var file = file
        var fileQuery = fileQuery

        switch format {
        case .csv, .libsvm:
            fileQuery.append("format=\(format)")
        case .binary:
            break
        }

        if let labelColumn = labelColumn {
            fileQuery.append("label_column=\(labelColumn)")
        }

        if useCache {
            if format != .libsvm {
                throw ValueError.runtimeError("Cache currently only support convert from libsvm file.")
            }

            file += "#\(name).cache"
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
            baseMargin: baseMargin,
            labelLowerBound: labelLowerBound,
            labelUpperBound: labelUpperBound
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

    /// Load previously saved feature map.
    ///
    /// - Parameter from: Path to the feature map.
    public func loadFeatureMap(
        from path: String
    ) throws {
        try set(features: try [Feature](fromFeatureMap: path))
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

    /// - Returns: The shape of dmatrix, i.e. (rowCount(), columnCount()).
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
    /// - Returns: A new dmatrix class containing only selected indexes.
    public func slice(
        indexes: Int32Data,
        newName: String? = nil,
        allowGroups: Bool = false
    ) throws -> DMatrix {
        let indexes = try indexes.data()
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

        return try DMatrix(
            name: newName ?? name,
            dmatrix: slicedDmatrix,
            features: _features,
            label: nil,
            weight: nil,
            baseMargin: nil,
            labelLowerBound: nil,
            labelUpperBound: nil
        )
    }

    /// Slice and return a new Data that only indexes from given range..
    ///
    /// - Parameter indexes: Range of indices to be selected.
    /// - Parameter newName: New name of the returned Data.
    /// - Parameter allowGroups: Allow slicing of a matrix with a groups attribute.
    /// - Returns: A new DMatrix class containing only selected indexes.
    public func slice(
        indexes: Range<Int32>,
        newName: String? = nil,
        allowGroups: Bool = false
    ) throws -> DMatrix {
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
        values: UInt32Data
    ) throws {
        let values = try values.data()
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
        values: FloatData
    ) throws {
        let values = try values.data()
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
        field: String
    ) throws -> [UInt32] {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafePointer<UInt32>?>.allocate(capacity: 1)

        try safe {
            XGDMatrixGetUIntInfo(dmatrix, field, outLenght, outResult)
        }

        return (0 ..< Int(outLenght.pointee)).lazy.map { outResult.pointee![$0] }
    }

    /// Get float property from the DMatrixHandle.
    ///
    /// - Parameter field: The field name of the information.
    /// - Returns: An array of float information of the data.
    public func getFloatInfo(
        field: String
    ) throws -> [Float] {
        let outLenght = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        let outResult = UnsafeMutablePointer<UnsafePointer<Float>?>.allocate(capacity: 1)

        try safe {
            XGDMatrixGetFloatInfo(dmatrix, field, outLenght, outResult)
        }

        return (0 ..< Int(outLenght.pointee)).lazy.map { outResult.pointee![$0] }
    }

    /// Set uint type property into the DMatrixHandle.
    ///
    /// - Parameter field: The field name of the information.
    /// - Parameter values: The array of data to be set.
    public func set(
        field: UIntField,
        values: UInt32Data
    ) throws {
        let values = try values.data()
        if field == .group, try values.reduce(0, +) != rowCount() {
            throw ValueError.runtimeError("The sum of groups must equal to the number of rows in the dmatrix.")
        }

        try setUIntInfo(field: field.rawValue, values: values)
    }

    /// Set float type property into the DMatrixHandle.
    ///
    /// - Parameter field: The field name of the information.
    /// - Parameter values: The array of data to be set.
    public func set(
        field: FloatField,
        values: FloatData
    ) throws {
        let values = try values.data()
        switch field {
        case .label, .weight, .labelLowerBound, .labelUpperBound:
            if try values.count != rowCount() {
                throw ValueError.runtimeError("The count of values must equal to the number of rows in the dmatrix.")
            }
        case .baseMargin:
            break
        }

        try setFloatInfo(field: field.rawValue, values: values)
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

    /// - Returns: Features of data, generates universal names with quantitative type if not previously set by user.
    public func features(defaultPrefix: String = "F-") throws -> [Feature] {
        if _features == nil {
            _features = try (0 ..< columnCount()).map { Feature(name: "\(defaultPrefix)\($0)", type: .quantitative) }
        }

        return _features!
    }

    /// - Returns: An array of float information of the data.
    public func get(field: FloatField) throws -> [Float] {
        try getFloatInfo(field: field.rawValue)
    }

    /// - Returns: An array of uint information of the data.
    public func get(field: UIntField) throws -> [UInt32] {
        try getUIntInfo(field: field.rawValue)
    }
}
