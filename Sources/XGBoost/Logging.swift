import Logging

/// Logger instance used for messages.
let logger = Logger(label: "swiftxgboost")

/// - Parameters: Message to log.
func log(_ message: String) {
    // TODO: Rabit logging.
    logger.info("\(message)")
}
