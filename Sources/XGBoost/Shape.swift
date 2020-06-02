import Foundation

/// Shape of Data.
public struct Shape: ExpressibleByArrayLiteral, Equatable {
    public let row: Int
    public let column: Int
    public typealias ArrayLiteralElement = Int

    public init(_ row: Int, _ column: Int) {
        self.row = row
        self.column = column
    }

    public init(row: Int, column: Int) {
        self.row = row
        self.column = column
    }

    public init(arrayLiteral elements: Int...) {
        precondition(elements.count == 2, "Invalid shape of input.")
        row = elements[0]
        column = elements[1]
    }
}
