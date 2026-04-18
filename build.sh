#!/bin/bash

# IndustryVisionKit Build Script
# 使用方法: ./build.sh [libtorch|onnxruntime|clean]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

print_usage() {
    echo "用法: $0 [命令]"
    echo ""
    echo "可用命令:"
    echo "  libtorch      - 使用 LibTorch 后端编译（推荐）"
    echo "  onnxruntime   - 使用 ONNX Runtime 后端编译"
    echo "  simulator     - 仅编译模拟器模式（无依赖）"
    echo "  clean         - 清除编译输出"
    echo "  run           - 编译并运行（使用上次的后端选择）"
    echo ""
    echo "示例:"
    echo "  $0 libtorch     # 使用 LibTorch 编译"
    echo "  $0 onnxruntime  # 使用 ONNX Runtime 编译"
    echo "  $0 run          # 编译并运行"
}

build_with_libtorch() {
    echo "🔨 使用 LibTorch 编译..."
    
    if [ ! -d "${PROJECT_ROOT}/libtorch" ]; then
        echo "❌ 错误: 找不到 libtorch 目录"
        echo "请确保 libtorch-macos-arm64-2.11.0.zip 已解压到 ${PROJECT_ROOT}"
        exit 1
    fi
    
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    cmake .. \
        -DINDUSTRYVISION_ENABLE_LIBTORCH=ON \
        -DINDUSTRYVISION_LIBTORCH_ROOT="${PROJECT_ROOT}/libtorch"
    
    make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    echo "✅ LibTorch 编译完成！"
    echo "运行: ${PROJECT_ROOT}/bin/IndustryVisionKit"
}

build_with_onnxruntime() {
    echo "🔨 使用 ONNX Runtime 编译..."
    
    if [ ! -d "${PROJECT_ROOT}/onnxruntime" ]; then
        echo "❌ 错误: 找不到 onnxruntime 目录"
        echo "请从以下位置下载 ONNX Runtime:"
        echo "https://github.com/microsoft/onnxruntime/releases"
        echo ""
        echo "然后解压到: ${PROJECT_ROOT}/onnxruntime"
        exit 1
    fi
    
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    cmake .. \
        -DINDUSTRYVISION_ENABLE_ONNXRUNTIME=ON \
        -DINDUSTRYVISION_ONNXRUNTIME_ROOT="${PROJECT_ROOT}/onnxruntime"
    
    make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    echo "✅ ONNX Runtime 编译完成！"
    echo "运行: ${PROJECT_ROOT}/bin/IndustryVisionKit"
}

build_simulator() {
    echo "🔨 编译模拟器模式..."
    
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    cmake ..
    make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    echo "✅ 模拟器模式编译完成！"
    echo "运行: ${PROJECT_ROOT}/bin/IndustryVisionKit"
}

clean_build() {
    echo "🧹 清除编译缓存..."
    rm -rf "${BUILD_DIR}"
    echo "✅ 清除完成"
}

run_application() {
    if [ ! -f "${PROJECT_ROOT}/bin/IndustryVisionKit" ]; then
        echo "⚠️  可执行文件不存在，先进行编译..."
        build_with_libtorch
    fi
    
    echo "🚀 启动应用..."
    
    export DYLD_LIBRARY_PATH="${PROJECT_ROOT}/libtorch/lib:$DYLD_LIBRARY_PATH"
    export DYLD_LIBRARY_PATH="${PROJECT_ROOT}/onnxruntime/lib:$DYLD_LIBRARY_PATH"
    
    "${PROJECT_ROOT}/bin/IndustryVisionKit"
}

# 主程序
case "${1:-}" in
    libtorch)
        build_with_libtorch
        ;;
    onnxruntime)
        build_with_onnxruntime
        ;;
    simulator)
        build_simulator
        ;;
    clean)
        clean_build
        ;;
    run)
        run_application
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
