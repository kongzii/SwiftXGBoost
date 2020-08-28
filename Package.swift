// swift-tools-version:5.1

import PackageDescription

let package = Package(
    name: "SwiftXGBoost",
    platforms: [
        .macOS(.v10_13),
    ],
    products: [
        .library(
            name: "XGBoost",
            targets: [
                "XGBoost",
            ]
        ),
        .library(
            name: "CXGBoost",
            targets: [
                "CXGBoost",
            ]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/KarthikRIyer/swiftplot.git", from: "2.0.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.4.0"),
    ],
    targets: [
        .target(
            name: "XGBoost",
            dependencies: [
                "CXGBoost",
                "Logging",
                "SwiftPlot",
                "SVGRenderer",
            ]
        ),
        .systemLibrary(
            name: "CXGBoost",
            pkgConfig: "xgboost",
            providers: [
                .brew(["xgboost"]),
            ]
        ),
        .testTarget(
            name: "XGBoostTests",
            dependencies: [
                "XGBoost",
            ]
        ),
        .target(
            name: "AftSurvival",
            dependencies: [
                "XGBoost",
            ],
            path: "Examples/AftSurvival"
        ),
        .target(
            name: "CrossValidation",
            dependencies: [
                "XGBoost",
            ],
            path: "Examples/CrossValidation"
        ),
        .target(
            name: "BoostFromPrediction",
            dependencies: [
                "XGBoost",
            ],
            path: "Examples/BoostFromPrediction"
        ),
        .target(
            name: "CustomObjective",
            dependencies: [
                "XGBoost",
            ],
            path: "Examples/CustomObjective"
        ),
        .target(
            name: "EvalsResult",
            dependencies: [
                "XGBoost",
            ],
            path: "Examples/EvalsResult"
        ),
        .target(
            name: "ExternalMemory",
            dependencies: [
                "XGBoost",
            ],
            path: "Examples/ExternalMemory"
        ),
        .target(
            name: "GeneralizedLinearModel",
            dependencies: [
                "XGBoost",
            ],
            path: "Examples/GeneralizedLinearModel"
        ),
        .target(
            name: "PredictFirstNTree",
            dependencies: [
                "XGBoost",
            ],
            path: "Examples/PredictFirstNTree"
        ),
        .target(
            name: "PredictLeafIndices",
            dependencies: [
                "XGBoost",
            ],
            path: "Examples/PredictLeafIndices"
        ),
    ]
)
