import Foundation

public
typealias Parameter = (name: String, value: String)

public
typealias Version = (major: Int, minor: Int, patch: Int)

public
typealias Feature = (name: String, type: FeatureType)

public
typealias BufferModel = (length: UInt64, data: UnsafeMutablePointer<UnsafePointer<Int8>?>)

public
typealias Shape = (row: Int, column: Int)

public enum AfterIterationCallbackOutput {
    case stop, next
}

public enum FeatureType: String {
    case float
    case int
    case quantitative = "q"
    case indicator = "i"
}

public enum ModelFormat: String {
    case text, json
}

public enum DataFormat: String {
    case binary, csv
}

public enum FloatField: String {
    case label
    case weight
    case baseMargin = "base_margin"
}

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

public enum Booster: String {
    case gbtree
    case gblinear
    case dart
}
