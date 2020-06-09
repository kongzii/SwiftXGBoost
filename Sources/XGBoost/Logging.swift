import Logging

let logger = Logger(label: "swiftxgboost")

func log(_ message: String) {
    // TODO: Rabit logging.
    logger.info("\(message)")
}
