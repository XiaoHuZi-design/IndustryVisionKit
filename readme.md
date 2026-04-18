# IndustryVisionKit

基于 **C++17 / Qt6 / CMake / ONNX Runtime** 的工业视觉检测桌面应用，支持 YOLOv5、YOLOv8、YOLO11、YOLO26 等多版本目标检测模型，提供图片、视频、摄像头三种输入模式。

## 功能特性

- 多版本 YOLO 模型推理（YOLOv5 / v8 / v11 / v26）
- 图片 / 视频 / 摄像头（含多设备选择、热插拔刷新）三种检测模式
- 实时视频流检测与标注
- 置信度、IOU 阈值实时调节
- 检测结果表格展示、汇总统计、CSV/TXT 导出
- 用户注册登录（本地 JSON 持久化）
- 运行日志实时输出
- 自定义类别标签文件支持

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | C++17 |
| GUI 框架 | Qt6 Widgets（纯代码构建，不使用 .ui） |
| 构建系统 | CMake 3.21+ |
| 推理引擎 | ONNX Runtime 1.24+ |
| 图像处理 | OpenCV 4.x |
| 模型格式 | ONNX (.onnx) |

## 项目结构

```
IndustryVisionKit/
├── CMakeLists.txt                  # 根 CMake 配置
├── IndustryVisionLib/              # 算法层
│   ├── CMakeLists.txt
│   ├── include/IndustryVisionLib/
│   │   ├── DetectionTypes.h        # 检测数据结构与配置
│   │   ├── UserManager.h           # 用户管理（注册/登录/持久化）
│   │   └── YoloEngine.h            # YOLO 推理引擎封装
│   └── src/
│       ├── UserManager.cpp
│       └── YoloEngine.cpp
├── IndustryVisionGUI/              # 界面层
│   ├── CMakeLists.txt
│   ├── include/IndustryVisionGUI/
│   │   ├── ApplicationWindow.h     # 主窗口（页面切换）
│   │   ├── LoginWidget.h           # 登录/注册页
│   │   └── DetectionWidget.h       # 检测主页
│   └── src/
│       ├── main.cpp
│       ├── ApplicationWindow.cpp
│       ├── LoginWidget.cpp
│       └── DetectionWidget.cpp
├── resource/                       # 资源文件（需自行准备，见下方说明）
│   ├── models/                     # ONNX 模型文件
│   ├── images/                     # 测试图片
│   └── classes/                    # 类别标签文件
├── doc/
│   └── coding_rules.md
└── README.md
```

## 环境依赖

### 必需

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| CMake | 3.21+ | 构建系统 |
| Qt | 6.0+ | 需包含 Core、Gui、Widgets 模块 |
| OpenCV | 4.0+ | 需包含 core、imgproc、imgcodecs、video、videoio、highgui |
| ONNX Runtime | 1.17+ | YOLO 模型推理引擎 |
| C++ 编译器 | C++17 支持 | macOS Clang 15+ / GCC 9+ / MSVC 2019+ |

### 可选

| 依赖 | 说明 |
|------|------|
| LibTorch | PyTorch C++ 后端（TorchScript 模型推理，已预留接口） |
| Python 3.12 + ultralytics | 用于将 PyTorch .pt 模型转换为 ONNX 格式 |

## 安装与构建

### 1. 安装依赖

**macOS (Homebrew)：**

```bash
# Qt6
brew install qt

# OpenCV
brew install opencv

# ONNX Runtime — 从 GitHub Releases 下载预编译包
# https://github.com/microsoft/onnxruntime/releases
# 下载 onnxruntime-osx-arm64-*.tgz，解压到项目根目录
```

**Ubuntu (apt)：**

```bash
sudo apt install cmake qt6-base-dev libopencv-dev
# ONNX Runtime 需从 GitHub 下载预编译包
```

**Windows (vcpkg)：**

```bash
vcpkg install qt6-base opencv4 onnxruntime
```

### 2. 准备模型文件

模型文件体积较大，不包含在 Git 仓库中。请将 ONNX 模型放到 `resource/models/` 目录：

```bash
mkdir -p resource/models
mkdir -p resource/images
mkdir -p resource/classes
```

推荐模型（从 [Ultralytics](https://github.com/ultralytics/ultralytics) 获取）：

| 模型 | 来源 | 默认类别文件 |
|------|------|-------------|
| yolov5s.onnx | YOLOv5 官方 | coco.names.txt |
| yolov8n.onnx | YOLOv8 官方 | coco.names.txt |
| yolo11n.onnx | YOLO11 官方 | coco.names.txt |
| yolo26n.onnx | YOLO26 官方 | coco.names.txt |

可用 Python 一键导出 ONNX：

```bash
pip install ultralytics onnx onnxslim

yolo export model=yolov8n.pt format=onnx
yolo export model=yolo11n.pt format=onnx
```

默认类别文件 `coco.names.txt`（80 类 COCO 数据集）已包含在 `resource/classes/` 中。自定义模型请创建对应的 `.txt` 类别文件。

### 3. 构建

```bash
# 配置（根据实际安装路径修改）
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="/path/to/Qt6/lib/cmake;/path/to/onnxruntime" \
  -DINDUSTRYVISION_ENABLE_ONNXRUNTIME=ON \
  -DINDUSTRYVISION_ONNXRUNTIME_ROOT=/path/to/onnxruntime

# 编译
cmake --build build --config Release -j$(nproc)
```

构建产物输出到 `bin/IndustryVisionKit`。

**macOS 示例（Homebrew 安装）：**

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$(brew --prefix qt)/lib/cmake;$(pwd)/onnxruntime-osx-arm64-1.24.4" \
  -DINDUSTRYVISION_ENABLE_ONNXRUNTIME=ON \
  -DINDUSTRYVISION_ONNXRUNTIME_ROOT=$(pwd)/onnxruntime-osx-arm64-1.24.4

cmake --build build -j$(sysctl -n hw.ncpu)
```

### 4. 运行

```bash
# macOS 需设置库路径
export DYLD_LIBRARY_PATH=/path/to/onnxruntime/lib:$DYLD_LIBRARY_PATH

./bin/IndustryVisionKit
```

## 使用说明

1. **注册登录** — 首次使用先注册账号，用户信息保存在本地
2. **选择模型** — 在左侧面板选择 YOLO 版本，系统自动匹配默认模型路径；也可手动浏览选择自定义模型
3. **调整参数** — 置信度和 IOU 阈值实时生效，无需手动确认
4. **选择输入源** — 支持图片/视频/摄像头模式；摄像头模式自动探测可用设备，支持刷新
5. **开始检测** — 点击"开始"执行检测，右侧显示原图和标注结果
6. **查看结果** — 下方表格显示检测详情，支持导出 CSV/TXT

## YOLO 版本输出格式说明

不同 YOLO 版本的 ONNX 输出格式有差异，本项目已适配：

| 版本 | 输出形状 | 说明 |
|------|---------|------|
| YOLOv5 | (1, 25200, 85) | 含 objectness 置信度 |
| YOLOv8/v11 | (1, 84, 8400) | 无 objectness，需转置 |
| YOLO26 E2E | (1, 300, 6) | 端到端格式，直接输出 [x1,y1,x2,y2,score,class] |

## 后续扩展

- [ ] LibTorch 后端支持（.pt / TorchScript 模型）
- [ ] TensorRT 加速推理
- [ ] 实例分割模型支持（YOLOv8-seg 等）
- [ ] 多模型并发切换
- [ ] 检测历史记录与统计
- [ ] 目标跟踪（DeepSORT / ByteTrack）

## License

MIT License
