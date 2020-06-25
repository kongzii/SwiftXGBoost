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
            let ndarray = numpy.ctypeslib.as_array(data, shape: shape)
            return numpy.copy(ndarray)
        }
    }
}

public extension ArrayWithShape where Element: NumpyScalarCompatible {
    /// - Precondition: The `numpy` Python package must be installed.
    /// - Returns: Properly shaped numpy array.
    func makeNumpyArray() -> PythonObject {
        array.withUnsafeBytes { bytes in
            let data = ctypes.cast(Int(bitPattern: bytes.baseAddress), ctypes.POINTER(ctypes.c_float))
            let ndarray = numpy.ctypeslib.as_array(data, shape: shape)
            return numpy.copy(ndarray)
        }
    }
}

extension Shape {
    /// Init shape from PythonObject.
    ///
    /// - Parameter shape: Python object holding integers that can be converted to [Int].
    public init(_ shape: PythonObject) {
        self = [Int](shape)!
    }   

    /// Init shape from PythonObjects.
    ///
    /// - Parameter shape: Python object holding integers that can be converted to [Int].
    public init(_ elements: PythonObject...) {
        self = elements.map { Int($0)! }
    }   
}

/// PythonObject comfortances for protocols that allows using python objects seamlessly with Booster and DMatrix.
extension PythonObject: FloatData, Int32Data, UInt32Data, ShapeData {
    /// Comfortance for FloatData.
    public func data() throws -> [Float] {
        if Bool(Python.isinstance(self, numpy.ndarray))! {
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

    /// Comfortance for Int32Data.
    public func data() throws -> [Int32] {
        if Bool(Python.isinstance(self, numpy.ndarray))! {
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

    /// Comfortance for UInt32Data.
    public func data() throws -> [UInt32] {
        if Bool(Python.isinstance(self, numpy.ndarray))! {
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

    /// Comfortance for ShapeData.
    public func dataShape() throws -> Shape {
        if Bool(Python.isinstance(self, numpy.ndarray))! {
            return Shape(self.shape)
        } else {
            throw ValueError.runtimeError("PythonObject type \(Python.type(self)) is not supported ShapeData.")
        }
    }
}
