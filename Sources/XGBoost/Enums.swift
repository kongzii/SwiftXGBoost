import Foundation

/// Typealias for underlying XGBoost version.
public typealias Version = (major: Int, minor: Int, patch: Int)

/// Tuple holding length of buffer along with it, so it can be easily read.
public typealias RawModel = (length: UInt64, data: UnsafeMutablePointer<UnsafePointer<Int8>?>)
public typealias BufferModel = (length: UInt64, data: UnsafeMutablePointer<UnsafePointer<Int8>?>)
public typealias SerializedBuffer = (length: UInt64, data: UnsafeMutablePointer<UnsafePointer<Int8>?>)

/// Indicates if iteration should break (stop) or continue (next).
public enum AfterIterationOutput {
    case stop, next
}

/// Currently supported model formats.
public enum ModelFormat: String {
    case text, json, dot
}

/// Currently supported data formats.
public enum DataFormat: String {
    case libsvm, csv, binary
}

/// Predefined names of float fields settable in DMatrix.
/// You can set also another fields using method accepting strings.
public enum FloatField: String {
    case label
    case weight
    case baseMargin = "base_margin"
    case labelLowerBound = "label_lower_bound"
    case labelUpperBound = "label_upper_bound"
}

/// Predefined names of uint fields settable in DMatrix.
/// You can set also another fields using method accepting strings.
public enum UIntField: String {
    case group
    case groupPtr = "group_ptr"
}

/// Supported types for importance graph.
public enum Importance: String {
    case weight
    case gain
    case cover
    case totalGain = "total_gain"
    case totalCover = "total_cover"
}

/// Supported Booster types.
public enum BoosterType: String {
    case gbtree
    case gblinear
    case dart
}
