#if canImport(TensorFlow)

    import TensorFlow

    extension Tensor: FloatData where Scalar == Float {
        /// Comfortance for FloatData.
        public func data() throws -> [Float] {
            scalars
        }
    }

    extension Tensor: UInt32Data where Scalar == UInt32 {
        /// Comfortance for UInt32Data.
        public func data() throws -> [UInt32] {
            scalars
        }
    }

    extension Tensor: ShapeData {
        /// Comfortance for ShapeData.
        public func dataShape() throws -> Shape {
            Shape(shape)
        }
    }

#endif
