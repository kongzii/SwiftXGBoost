#if canImport(TensorFlow)

    import TensorFlow

    extension Tensor: FloatData, DMatrixShape where Scalar == Float {
        public func data() throws -> [Float] {
            scalars
        }

        public func dataShape() throws -> Shape {
            if shape.count == 1 {
                return try Shape(row: 1, column: shape[0])
            } else if shape.count == 2 {
                return try Shape(row: shape[0], column: shape[1])
            } else {
                throw ValueError.runtimeError("Invalid shape \(shape) of self.")
            }
        }
    }

#endif
