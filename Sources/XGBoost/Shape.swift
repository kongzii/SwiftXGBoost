import Foundation

/// Shape of data structure.
public struct Shape: ExpressibleByArrayLiteral, Equatable {
    public let row: Int
    public let column: Int
    public typealias ArrayLiteralElement = Int

    /// - Parameter: Number of rows.
    /// - Parameter: Number of columns.
    public init(_ row: Int, _ column: Int) {
        self.row = row
        self.column = column
    }

    /// - Parameter row: Number of rows.
    /// - Parameter column: Number of columns.
    public init(row: Int, column: Int) {
        self.row = row
        self.column = column
    }

    /// Initialize from array literal having two elements.
    /// - Precondition: arrayLiteral.coutn == 2
    public init(arrayLiteral elements: Int...) {
        precondition(elements.count == 2, "Invalid shape of input.")
        row = elements[0]
        column = elements[1]
    }
}
