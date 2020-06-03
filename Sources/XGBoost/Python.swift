import PythonKit

/// The `numpy` Python module.
/// Note: Global variables are lazy, so the following declaration won't produce
/// a Python import error until it is first used.
private let numpy = Python.import("numpy")
private let ctypes = Python.import("ctypes")

public extension Array where Element: NumpyScalarCompatible {
    /// - Precondition: The `numpy` Python package must be installed.
    /// - Returns: Numpy array of shape.
    func makeNumpyArray(shape: Shape) -> PythonObject {
        withUnsafeBytes { bytes in
            let data = ctypes.cast(Int(bitPattern: bytes.baseAddress), ctypes.POINTER(ctypes.c_float))
            let ndarray = numpy.ctypeslib.as_array(data, shape: [shape.row, shape.column])
            return numpy.copy(ndarray)
        }
    }
}

extension Shape {
    public init(_ row: PythonObject, _ column: PythonObject) throws {
        guard let row = Int(row), let column = Int(column) else {
            throw ValueError.runtimeError("Invalid type of python input.")
        }

        self.row = row
        self.column = column
    }
}

extension DMatrix {
    /// Initialize Data from python object array.
    /// Currently supported: Numpy NDArray.
    ///
    /// - Parameter name: Name of dataset.
    /// - Parameter from: Python object.
    /// - Parameter label: Array of labels for data.
    /// - Parameter weight: Weight for each instance.
    /// - Parameter baseMargin: Set base margin of booster to start from.
    /// - Parameter features: Names and types of features.
    /// - Parameter missingValue: Value in the input data which needs to be present as a missing value.
    /// - Parameter threads:  Number of threads to use for loading data when parallelization is applicable. If 0, uses maximum threads available on the system.
    public convenience init(
        name: String,
        from array: PythonObject,
        label: [Float]? = nil,
        weight: [Float]? = nil,
        baseMargin: [Float]? = nil,
        features: [Feature]? = nil,
        missingValue: Float = Float.greatestFiniteMagnitude,
        threads: Int = 0
    ) throws {
        if !Bool(Python.isinstance(array, numpy.ndarray))! {
            throw ValueError.runtimeError("PythonObject is not a numpy ndarray.")
        } else if array.shape.count != 2 {
            throw ValueError.runtimeError("Invalid shape \(array.shape) of input.")
        }

        let data = numpy.array(array.reshape(array.size), copy: false, dtype: numpy.float32)
        let contiguousData = numpy.ascontiguousarray(data)

        guard let ptrVal = UInt(contiguousData.__array_interface__["data"].tuple2.0) else {
            throw ValueError.runtimeError("Can not get pointer value from numpy object.")
        }

        guard let pointer = UnsafePointer<Float>(bitPattern: ptrVal) else {
            throw ValueError.runtimeError("numpy.ndarray data pointer was nil.")
        }

        try self.init(
            name: name,
            from: pointer,
            shape: Shape(array.shape[0], array.shape[1]),
            label: label,
            weight: weight,
            baseMargin: baseMargin,
            features: features,
            missingValue: missingValue,
            threads: threads
        )
    }

    /// Initialize Data from python object array.
    /// Currently supported: Numpy NDArray.
    ///
    /// - Parameter name: Name of dataset.
    /// - Parameter from: Python object.
    /// - Parameter label: Python object.
    /// - Parameter weight: Weight for each instance.
    /// - Parameter baseMargin: Set base margin of booster to start from.
    /// - Parameter features: Names and types of features.
    /// - Parameter missingValue: Value in the input data which needs to be present as a missing value.
    /// - Parameter threads:  Number of threads to use for loading data when parallelization is applicable. If 0, uses maximum threads available on the system.
    public convenience init(
        name: String,
        from: PythonObject,
        label: PythonObject,
        weight: [Float]? = nil,
        baseMargin: [Float]? = nil,
        features: [Feature]? = nil,
        missingValue: Float = Float.greatestFiniteMagnitude,
        threads: Int = 0
    ) throws {
        if !Bool(Python.isinstance(label, numpy.ndarray))! {
            throw ValueError.runtimeError("PythonObject is not a numpy ndarray.")

            if label.shape.count != 1 {
                throw ValueError.runtimeError("Invalid shape \(label.shape) of label.")
            }
        }

        try self.init(
            name: name,
            from: from,
            label: [Float](label)!,
            weight: weight,
            baseMargin: baseMargin,
            features: features,
            missingValue: missingValue,
            threads: threads
        )
    }
}
