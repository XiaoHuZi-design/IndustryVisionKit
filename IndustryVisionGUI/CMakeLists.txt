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
