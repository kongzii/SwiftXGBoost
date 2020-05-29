// swift-tools-version:5.1

import PackageDescription

let package = Package(
    name: "SwiftXGBoost",
    platforms: [
        .macOS(.v10_15),
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
        .package(url: "https://github.com/pvieito/PythonKit.git", .branch("master")),
        .package(url: "https://github.com/KarthikRIyer/swiftplot.git", from: "2.0.0"),
    ],
    targets: [
        .target(
            name: "XGBoost",
            dependencies: [
                "CXGBoost",
                "SwiftPlot",
                "AGGRenderer",
            ],
            cSettings: [
                .unsafeFlags(["-I/usr/local/include"]),
            ],
            linkerSettings: [
                .unsafeFlags(["-L/usr/local/lib"]),
            ]
        ),
        .systemLibrary(
            name: "CXGBoost",
            providers: [
                .brew(["xgboost"]),
            ]
        ),
        .testTarget(
            name: "XGBoostTests",
            dependencies: [
                "XGBoost",
                "PythonKit",
            ]
        ),
    ]
)
