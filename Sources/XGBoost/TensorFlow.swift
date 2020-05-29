#if canImport(TensorFlow)

    import TensorFlow

    extension Data {
        /// Initialize Data from Tensor.
        ///
        /// - Parameter name: Name of dataset.
        /// - Parameter from: 2D Tensor to get values from.
        /// - Parameter label: Array of labels for data.
        /// - Parameter weight: Weight for each instance.
        /// - Parameter baseMargin: Set base margin of booster to start from.
        /// - Parameter features: Names and types of features.
        /// - Parameter missingValue: Value in the input data which needs to be present as a missing value.
        /// - Parameter threads:  Number of threads to use for loading data when parallelization is applicable. If 0, uses maximum threads available on the system.
        public convenience init(
            name: String,
            from values: Tensor<Float>,
            label: [Float]? = nil,
            weight: [Float]? = nil,
            baseMargin: [Float]? = nil,
            features: [Feature]? = nil,
            missingValue: Float = Float.greatestFiniteMagnitude,
            threads: Int = 0
        ) throws {
            if values.rank != 2 {
                throw ValueError.runtimeError("Invalid shape \(values.shape) of tensor. Shape must be (rows, columns).")
            }

            try self.init(
                name: name,
                values: values.scalars,
                shape: (values.shape[0], values.shape[1]),
                label: label,
                weight: weight,
                baseMargin: baseMargin,
                features: features,
                missingValue: missingValue,
                threads: threads
            )
        }
    }

    extension Tensor where Scalar == Float {
        /// Convert Tensor to Data.
        ///
        /// - Parameter name: Name of dataset.
        /// - Parameter label: Array of labels for data.
        /// - Parameter weight: Weight for each instance.
        /// - Parameter baseMargin: Set base margin of booster to start from.
        /// - Parameter features: Names and types of features.
        /// - Parameter missingValue: Value in the input data which needs to be present as a missing value.
        /// - Parameter threads:  Number of threads to use for loading data when parallelization is applicable. If 0, uses maximum threads available on the system.
        /// - Return: Data from tensor.
        public func toXGBoostData(
            name: String,
            label: [Float]? = nil,
            weight: [Float]? = nil,
            baseMargin: [Float]? = nil,
            features: [Feature]? = nil,
            missingValue: Float = Float.greatestFiniteMagnitude,
            threads: Int = 0
        ) throws -> Data {
            try Data(
                name: name,
                from: self,
                label: label,
                weight: weight,
                baseMargin: baseMargin,
                features: features,
                missingValue: missingValue,
                threads: threads
            )
        }
    }

#endif
