find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui)

add_library(IndustryVisionLib STATIC
    include/IndustryVisionLib/DetectionTypes.h
    include/IndustryVisionLib/InferenceBackend.h
    include/IndustryVisionLib/UserManager.h
    include/IndustryVisionLib/YoloEngine.h
    src/BackendOpenCV.cpp
    src/UserManager.cpp
    src/YoloEngine.cpp
)

# ONNX Runtime 后端（独立编译单元）
if(INDUSTRYVISION_ENABLE_ONNXRUNTIME)
    target_sources(IndustryVisionLib PRIVATE src/BackendOnnxRuntime.cpp)
endif()

# OpenVINO 后端（独立编译单元）
if(INDUSTRYVISION_ENABLE_OPENVINO)
    target_sources(IndustryVisionLib PRIVATE src/BackendOpenVINO.cpp)
endif()

# LibTorch 后端（独立编译单元）
if(INDUSTRYVISION_ENABLE_LIBTORCH)
    target_sources(IndustryVisionLib PRIVATE src/BackendLibTorch.cpp)
endif()

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

if(INDUSTRYVISION_ENABLE_OPENVINO)
    if(NOT INDUSTRYVISION_OPENVINO_ROOT)
        message(FATAL_ERROR "INDUSTRYVISION_ENABLE_OPENVINO=ON 时必须设置 INDUSTRYVISION_OPENVINO_ROOT")
    endif()

    set(OpenVINO_DIR ${INDUSTRYVISION_OPENVINO_ROOT}/runtime/cmake)
    find_package(OpenVINO REQUIRED)

    target_compile_definitions(IndustryVisionLib
        PRIVATE
            INDUSTRYVISION_HAS_OPENVINO
    )

    target_link_libraries(IndustryVisionLib
        PRIVATE
            openvino::runtime
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
        PUBLIC
            ${INDUSTRYVISION_LIBTORCH_INCLUDE_DIR}
            ${INDUSTRYVISION_LIBTORCH_ROOT}/include/torch/csrc/api/include
    )

    target_compile_definitions(IndustryVisionLib
        PUBLIC
            INDUSTRYVISION_HAS_LIBTORCH
    )

    # LibTorch 需要链接完整依赖链（静态库方式）
    find_library(INDUSTRYVISION_C10_LIBRARY NAMES c10 PATHS ${INDUSTRYVISION_LIBTORCH_LIB_DIR} REQUIRED)
    find_library(INDUSTRYVISION_TORCH_GLOBAL_DEPS_LIBRARY NAMES torch_global_deps PATHS ${INDUSTRYVISION_LIBTORCH_LIB_DIR} REQUIRED NO_DEFAULT_PATH)

    target_link_libraries(IndustryVisionLib
        PUBLIC
            ${INDUSTRYVISION_TORCH_LIBRARY}
            ${INDUSTRYVISION_TORCH_CPU_LIBRARY}
            ${INDUSTRYVISION_C10_LIBRARY}
            ${INDUSTRYVISION_LIBTORCH_LIB_DIR}/libtorch_global_deps.dylib
    )

    # LibTorch 运行时需要找到动态库
    set(INDUSTRYVISION_LIBTORCH_LIB_DIR ${INDUSTRYVISION_LIBTORCH_ROOT}/lib PARENT_SCOPE)
endif()

target_link_libraries(IndustryVisionLib
    PUBLIC
        Qt${QT_VERSION_MAJOR}::Core
        Qt${QT_VERSION_MAJOR}::Gui
        opencv_dnn
        opencv_core
)
