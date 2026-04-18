find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui)

add_library(IndustryVisionLib STATIC
    include/IndustryVisionLib/DetectionTypes.h
    include/IndustryVisionLib/UserManager.h
    include/IndustryVisionLib/YoloEngine.h
    src/UserManager.cpp
    src/YoloEngine.cpp
)

target_include_directories(IndustryVisionLib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(INDUSTRYVISION_ENABLE_ONNXRUNTIME)
    if(NOT INDUSTRYVISION_ONNXRUNTIME_ROOT)
        message(FATAL_ERROR "INDUSTRYVISION_ENABLE_ONNXRUNTIME=ON 时必须设置 INDUSTRYVISION_ONNXRUNTIME_ROOT")
    endif()

    set(INDUSTRYVISION_ONNXRUNTIME_INCLUDE_DIR ${INDUSTRYVISION_ONNXRUNTIME_ROOT}/include)
    find_library(INDUSTRYVISION_ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS
            ${INDUSTRYVISION_ONNXRUNTIME_ROOT}/lib
            ${INDUSTRYVISION_ONNXRUNTIME_ROOT}/lib64
        REQUIRED
    )

    target_include_directories(IndustryVisionLib
        PRIVATE
            ${INDUSTRYVISION_ONNXRUNTIME_INCLUDE_DIR}
    )

    target_compile_definitions(IndustryVisionLib
        PRIVATE
            INDUSTRYVISION_HAS_ONNXRUNTIME
    )

    target_link_libraries(IndustryVisionLib
        PRIVATE
            ${INDUSTRYVISION_ONNXRUNTIME_LIBRARY}
    )
endif()

if(INDUSTRYVISION_ENABLE_LIBTORCH)
    if(NOT INDUSTRYVISION_LIBTORCH_ROOT)
        message(FATAL_ERROR "INDUSTRYVISION_ENABLE_LIBTORCH=ON 时必须设置 INDUSTRYVISION_LIBTORCH_ROOT")
    endif()

    set(INDUSTRYVISION_LIBTORCH_INCLUDE_DIR ${INDUSTRYVISION_LIBTORCH_ROOT}/include)
    set(INDUSTRYVISION_LIBTORCH_LIB_DIR ${INDUSTRYVISION_LIBTORCH_ROOT}/lib)

    find_library(INDUSTRYVISION_TORCH_LIBRARY
        NAMES torch
        PATHS ${INDUSTRYVISION_LIBTORCH_LIB_DIR}
        REQUIRED
    )

    find_library(INDUSTRYVISION_TORCH_CPU_LIBRARY
        NAMES torch_cpu
        PATHS ${INDUSTRYVISION_LIBTORCH_LIB_DIR}
        REQUIRED
    )

    target_include_directories(IndustryVisionLib
        PRIVATE
            ${INDUSTRYVISION_LIBTORCH_INCLUDE_DIR}
            ${INDUSTRYVISION_LIBTORCH_ROOT}/include/torch/csrc/api/include
    )

    target_compile_definitions(IndustryVisionLib
        PRIVATE
            INDUSTRYVISION_HAS_LIBTORCH
    )

    target_link_libraries(IndustryVisionLib
        PRIVATE
            ${INDUSTRYVISION_TORCH_LIBRARY}
            ${INDUSTRYVISION_TORCH_CPU_LIBRARY}
    )
endif()

target_link_libraries(IndustryVisionLib
    PUBLIC
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
)
