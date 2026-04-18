find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Widgets)

add_executable(IndustryVisionKit
    include/IndustryVisionGUI/ApplicationWindow.h
    include/IndustryVisionGUI/LoginWidget.h
    include/IndustryVisionGUI/DetectionWidget.h
    src/main.cpp
    src/ApplicationWindow.cpp
    src/LoginWidget.cpp
    src/DetectionWidget.cpp
)

target_include_directories(IndustryVisionKit
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(IndustryVisionKit
    PRIVATE
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Widgets
        IndustryVisionLib
        opencv_core opencv_imgproc opencv_imgcodecs opencv_video opencv_videoio opencv_highgui
)

# 设置运行时库搜索路径（各后端动态库）
set(EXTRA_RPATHS
    "${CMAKE_SOURCE_DIR}/libtorch/lib"
    "${CMAKE_SOURCE_DIR}/openvino_runtime/runtime/lib/arm64/Release"
)
set_target_properties(IndustryVisionKit PROPERTIES
    BUILD_RPATH "${EXTRA_RPATHS}"
    INSTALL_RPATH "@executable_path/../libtorch/lib;@executable_path/../openvino_runtime/runtime/lib/arm64/Release"
)
