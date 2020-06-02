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
    ],
    targets: [
        .target(
            name: "XGBoost",
            dependencies: [
                "CXGBoost",
                "SwiftPlot",
                "AGGRenderer",
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
    ]
)
