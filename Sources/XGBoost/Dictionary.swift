public extension Dictionary {
    /// - Parameter key: Key to retrieve from self.
    /// - Parameter or: Value that will be set for `key` if `self[key]` is nil. 
    /// - Returns: Value for `key` or `or` if `key` is not set.
    subscript(key: Key, or def: Value) -> Value {
        mutating get {
            self[key] ?? {
                self[key] = def
                return def
            }()
        }
        set { self[key] = newValue }
    }
}
