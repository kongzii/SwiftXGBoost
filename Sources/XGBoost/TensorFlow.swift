#if canImport(TensorFlow)

    import TensorFlow

    extension Tensor: FloatData where Scalar == Float {
        public func data() throws -> [Float] {
            scalars
        }
    }

    extension Tensor: UInt32Data where Scalar == UInt32 {
        public func data() throws -> [UInt32] {
            scalars
        }
    }

    extension Tensor: ShapeData {
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
