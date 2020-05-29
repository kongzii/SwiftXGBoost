import Foundation

public extension String {
    /// String in C-compatible format, passable to the C-functions.
    var cCompatible: UnsafePointer<Int8>? {
        withCString { (cString) -> UnsafePointer<Int8>? in
            let length = strlen(cString) + 1
            let name: UnsafeMutablePointer<Int8>? = .allocate(capacity: length)
            name!.initialize(from: cString, count: length)
            return UnsafePointer<Int8>(name)
        }
    }
}
