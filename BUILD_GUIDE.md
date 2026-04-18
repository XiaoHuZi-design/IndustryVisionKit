# IndustryVisionKit 编译配置指南

## 概述

IndustryVisionKit 支持多个推理后端：
- **Simulator Mode**（模拟模式）- 不需要任何依赖，快速测试
- **LibTorch Backend** - 用于 PyTorch 模型，已配置
- **ONNX Runtime Backend** - 用于 ONNX 模型

## LibTorch 配置（优先推荐）

### 1. LibTorch 已自动解压
文件位置：`libtorch/` （位于项目根目录）

### 2. 使用 LibTorch 编译

```bash
cd /Users/teng/QtProjects/ht/IndustryVisionKit
mkdir -p build && cd build

cmake .. \
    -DINDUSTRYVISION_ENABLE_LIBTORCH=ON \
    -DINDUSTRYVISION_LIBTORCH_ROOT=/Users/teng/QtProjects/ht/IndustryVisionKit/libtorch

make -j$(sysctl -n hw.ncpu)
```

**支持的设备**：
- ✅ Apple Silicon (M1/M2/M3+ 使用 MPS 加速)
- ✅ Intel Mac (使用 CPU 推理)
- ✅ Linux with CUDA (自动检测)

### 3. 编译成功后

可执行文件生成在：`bin/IndustryVisionKit`

## ONNX Runtime 配置

### 1. 下载 ONNX Runtime

从官网下载 macOS 版本：
https://github.com/microsoft/onnxruntime/releases

选择对应版本：
- **Apple Silicon**: `onnxruntime-osx-arm64-*.tar.gz`
- **Intel Mac**: `onnxruntime-osx-x86_64-*.tar.gz`

### 2. 解压到项目目录

```bash
cd /Users/teng/QtProjects/ht/IndustryVisionKit
tar -xzf onnxruntime-osx-arm64-*.tar.gz
mv onnxruntime-osx-arm64-* onnxruntime
```

### 3. 使用 ONNX Runtime 编译

```bash
cd /Users/teng/QtProjects/ht/IndustryVisionKit
mkdir -p build && cd build

cmake .. \
    -DINDUSTRYVISION_ENABLE_ONNXRUNTIME=ON \
    -DINDUSTRYVISION_ONNXRUNTIME_ROOT=/Users/teng/QtProjects/ht/IndustryVisionKit/onnxruntime

make -j$(sysctl -n hw.ncpu)
```

## 运行应用

```bash
./bin/IndustryVisionKit
```

系统会自动检测并使用最优推理后端。

## 故障排除

### 编译时 LibTorch 找不到

确保路径正确：
```bash
ls -la /Users/teng/QtProjects/ht/IndustryVisionKit/libtorch/
# 应该包含: include/, lib/, share/ 等目录
```

### 运行时报错：Library not found

检查库路径设置，运行时可能需要：
```bash
export DYLD_LIBRARY_PATH=/Users/teng/QtProjects/ht/IndustryVisionKit/libtorch/lib:$DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/Users/teng/QtProjects/ht/IndustryVisionKit/onnxruntime/lib:$DYLD_LIBRARY_PATH
./bin/IndustryVisionKit
```

### 或创建启动脚本

编辑 `run.sh`：
```bash
#!/bin/bash
export DYLD_LIBRARY_PATH="/Users/teng/QtProjects/ht/IndustryVisionKit/libtorch/lib:$DYLD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="/Users/teng/QtProjects/ht/IndustryVisionKit/onnxruntime/lib:$DYLD_LIBRARY_PATH"
./bin/IndustryVisionKit
```

## GUI 改进说明

### 本次更新内容

✅ **登陆界面优化**
- 窗口尺寸从 420x360 缩小至 350x280，更加紧凑
- 布局更加整洁

✅ **检测界面优化**
- 预览区域最小尺寸调整为 600x360，更灵活的缩放
- 预览标签添加了 `QSizePolicy::Expanding` 支持动态伸缩
- 顶部内容区与底部日志区比例调整为 4:2，更均衡
- 检测结果区与日志区比例为 2:1，优先展示结果
- 移除了固定高度限制，改用最大高度 80px，提高响应性

✅ **缩放和谐性**
- 所有窗口现在支持平滑缩放无冲突
- 登录后窗口尺寸调整为 1400x900，更好的布局空间

## 推荐使用方式

1. **快速测试**：
   ```bash
   cd build && make && cd .. && ./bin/IndustryVisionKit
   ```
   会自动进入 Simulator 模式

2. **启用 LibTorch 加速**（4 行命令）：
   ```bash
   cd build && rm -rf *
   cmake .. -DINDUSTRYVISION_ENABLE_LIBTORCH=ON -DINDUSTRYVISION_LIBTORCH_ROOT=${PWD}/../libtorch
   make -j8
   ```

## 常见问题

**Q: 如何切换 PyTorch 模型格式？**
在 GUI 上选择模型文件时，支持 `.pt` 和 `.pth` 格式的 JIT 导出模型。

**Q: ONNX 和 LibTorch 哪个更好？**
- LibTorch：性能更优（特别是在 Apple Silicon），支持自动设备检测
- ONNX Runtime：模型兼容性更好，支持多框架

**Q: 能同时启用两个后端吗？**
可以，但会优先使用 LibTorch。可修改 CMakeLists.txt 的优先级。

