import Foundation

/// Parameter for XGBoost.
/// See https://xgboost.readthedocs.io/en/latest/parameter.html for available parameters.
public typealias Parameter = (name: String, value: String)

/// Typealias for underlying XGBoost version.
public typealias Version = (major: Int, minor: Int, patch: Int)

/// Information about feature.
public typealias Feature = (name: String, type: FeatureType)

/// Tuple holding length of buffer along with it, so it can be easily read.
public typealias BufferModel = (length: UInt64, data: UnsafeMutablePointer<UnsafePointer<Int8>?>)

/// Shape of Data.
public typealias Shape = (row: Int, column: Int)

public enum AfterIterationCallbackOutput {
    case stop, next
}

/// Type of feature.
public enum FeatureType: String {
    case quantitative = "q"
    case indicator = "i"
}

/// Currently supported model formats.
public enum ModelFormat: String {
    case text, json
}

/// Currently supported data formats.
public enum DataFormat: String {
    case binary, csv
}

/// Predefined names of float fields settable in Data.
/// You can set also another fields using method accepting strings.
public enum FloatField: String {
    case label
    case weight
    case baseMargin = "base_margin"
}

/// Predefined names of uint fields settable in Data.
/// You can set also another fields using method accepting strings.
public enum UIntField: String {
    case group
}

public enum Importance: String {
    case weight
    case gain
    case cover
    case totalGain = "total_gain"
    case totalCover = "total_cover"
}

/// Booster types supported by XGBoost
public enum Booster: String {
    case gbtree
    case gblinear
    case dart
}
