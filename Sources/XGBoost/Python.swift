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
    public init(shape: PythonObject) throws {
        guard let row = Int(shape[0]), let column = Int(shape[1]) else {
            throw ValueError.runtimeError("Invalid type of python input.")
        }

        self.row = row
        self.column = column
    }

    public init(_ shape: PythonObject) throws {
        try self.init(shape: shape)
    }

    public init(row: PythonObject, column: PythonObject) throws {
        guard let row = Int(row), let column = Int(column) else {
            throw ValueError.runtimeError("Invalid type of python input.")
        }

        self.row = row
        self.column = column
    }

    public init(_ row: PythonObject, _ column: PythonObject) throws {
        try self.init(row: row, column: column)
    }
}

extension PythonObject: FloatData, Int32Data, UInt32Data, ShapeData {
    public func data() throws -> [Float] {
        if Bool(Python.isinstance(self, numpy.ndarray))! {
            if self.shape.count == 1 {
                return [Float](self)!
            }

            if self.shape.count != 2 {
                throw ValueError.runtimeError("Invalid shape \(self.shape) of self.")
            }

            let size = Int(self.size)!
            let data = numpy.array(self.reshape(size), copy: false, dtype: numpy.float32)
            let contiguousData = numpy.ascontiguousarray(data)

            guard let ptrVal = UInt(contiguousData.__array_interface__["data"].tuple2.0) else {
                throw ValueError.runtimeError("Can not get pointer value from numpy object.")
            }

            guard let pointer = UnsafePointer<Float>(bitPattern: ptrVal) else {
                throw ValueError.runtimeError("numpy.ndarray data pointer was nil.")
            }

            return Array(UnsafeBufferPointer(start: pointer, count: size))
        } else {
            throw ValueError.runtimeError("PythonObject type \(Python.type(self)) [\(self.dtype)] is not supported FloatData.")
        }
    }

    public func data() throws -> [Int32] {
        if Bool(Python.isinstance(self, numpy.ndarray))! {
            if self.shape.count == 1 {
                return [Int32](self)!
            }

            if self.shape.count != 2 {
                throw ValueError.runtimeError("Invalid shape \(self.shape) of self.")
            }

            let size = Int(self.size)!
            let data = numpy.array(self.reshape(size), copy: false, dtype: numpy.int32)
            let contiguousData = numpy.ascontiguousarray(data)

            guard let ptrVal = UInt(contiguousData.__array_interface__["data"].tuple2.0) else {
                throw ValueError.runtimeError("Can not get pointer value from numpy object.")
            }

            guard let pointer = UnsafePointer<Int32>(bitPattern: ptrVal) else {
                throw ValueError.runtimeError("numpy.ndarray data pointer was nil.")
            }

            return Array(UnsafeBufferPointer(start: pointer, count: size))
        } else {
            throw ValueError.runtimeError("PythonObject type \(Python.type(self)) [\(self.dtype)] is not supported Int32Data.")
        }
    }

    public func data() throws -> [UInt32] {
        if Bool(Python.isinstance(self, numpy.ndarray))! {
            if self.shape.count == 1 {
                return [UInt32](self)!
            }

            if self.shape.count != 2 {
                throw ValueError.runtimeError("Invalid shape \(self.shape) of self.")
            }

            let size = Int(self.size)!
            let data = numpy.array(self.reshape(size), copy: false, dtype: numpy.uint32)
            let contiguousData = numpy.ascontiguousarray(data)

            guard let ptrVal = UInt(contiguousData.__array_interface__["data"].tuple2.0) else {
                throw ValueError.runtimeError("Can not get pointer value from numpy object.")
            }

            guard let pointer = UnsafePointer<UInt32>(bitPattern: ptrVal) else {
                throw ValueError.runtimeError("numpy.ndarray data pointer was nil.")
            }

            return Array(UnsafeBufferPointer(start: pointer, count: size))
        } else {
            throw ValueError.runtimeError("PythonObject type \(Python.type(self)) [\(self.dtype)] is not supported UInt32Data.")
        }
    }

    public func dataShape() throws -> Shape {
        if Bool(Python.isinstance(self, numpy.ndarray))! {
            if self.shape.count == 1 {
                return try Shape(row: 1, column: self.shape[0])
            } else if self.shape.count == 2 {
                return try Shape(shape: self.shape)
            } else {
                throw ValueError.runtimeError("Invalid shape \(self.shape) of self.")
            }
        } else {
            throw ValueError.runtimeError("PythonObject type \(Python.type(self)) is not supported ShapeData.")
        }
    }
}
