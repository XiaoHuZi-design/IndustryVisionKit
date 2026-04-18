# 快速开始指南

## 🚀 3 步启动应用

### 方案 1: 使用 LibTorch（推荐 - 性能最优）

```bash
cd /Users/teng/QtProjects/ht/IndustryVisionKit
./build.sh libtorch
./bin/IndustryVisionKit
```

**3 秒完成！** LibTorch 已自动解压，无需额外下载。

---

### 方案 2: 使用 ONNX Runtime

1️⃣ **下载 ONNX Runtime**
```bash
# 访问官网下载对应版本
# Apple Silicon: onnxruntime-osx-arm64-*.tar.gz
# https://github.com/microsoft/onnxruntime/releases
```

2️⃣ **解压到项目目录**
```bash
cd /Users/teng/QtProjects/ht/IndustryVisionKit
tar -xzf ~/Downloads/onnxruntime-osx-arm64-*.tar.gz
mv onnxruntime-osx-arm64-* onnxruntime
```

3️⃣ **编译并运行**
```bash
./build.sh onnxruntime
./bin/IndustryVisionKit
```

---

### 方案 3: 仅模拟器模式（无依赖，立即测试）

```bash
./build.sh simulator
./bin/IndustryVisionKit
```

---

## 📋 脚本命令速查

| 命令 | 说明 |
|------|------|
| `./build.sh libtorch` | 使用 LibTorch 编译（推荐） |
| `./build.sh onnxruntime` | 使用 ONNX Runtime 编译 |
| `./build.sh simulator` | 仅编译模拟器模式 |
| `./build.sh run` | 编译并直接运行 |
| `./build.sh clean` | 清除编译缓存 |

---

## ✨ 本次 GUI 优化总结

### 登陆界面
- ✅ 窗口从 420x360 缩小至 350x280
- ✅ 更紧凑的布局，加载更快

### 检测界面  
- ✅ 预览区域支持更灵活的缩放（最小 600x360）
- ✅ 内容区与日志区比例优化（4:2）
- ✅ 检测结果与日志区比例调整（2:1，优先结果）
- ✅ 移除固定高度，改用最大高度，响应更流畅

### 整体缩放
- ✅ 登陆时窗口 350x280
- ✅ 检测时窗口 1400x900
- ✅ 缩放无冲突、无重叠

---

## 🔧 故障排除

### "找不到 libtorch 目录"
确保项目根目录有 `libtorch/` 文件夹：
```bash
ls -la /Users/teng/QtProjects/ht/IndustryVisionKit/libtorch/
```

### "找不到 onnxruntime 目录"
从官网下载并解压到项目目录：
```bash
cd /Users/teng/QtProjects/ht/IndustryVisionKit
# 下载后解压，重命名为 onnxruntime
ls -la ./onnxruntime/
```

### 运行时库加载失败
尝试手动设置库路径：
```bash
export DYLD_LIBRARY_PATH="/Volumes/path/to/IndustryVisionKit/libtorch/lib:$DYLD_LIBRARY_PATH"
./bin/IndustryVisionKit
```

---

## 📝 项目文件说明

```
IndustryVisionKit/
├── build.sh                    ← 编译脚本
├── BUILD_GUIDE.md              ← 详细编译指南
├── QUICK_START.md              ← 本文件
├── CMakeLists.txt              ← 主配置（支持 LibTorch + ONNX Runtime）
├── IndustryVisionLib/
│   ├── CMakeLists.txt          ← 库配置
│   ├── src/YoloEngine.cpp      ← 推理引擎实现
│   └── include/
├── IndustryVisionGUI/
│   ├── src/ApplicationWindow.cpp ← GUI 窗口管理
│   ├── src/DetectionWidget.cpp   ← 检测界面（已优化）
│   ├── src/LoginWidget.cpp       ← 登陆界面（已优化）
│   └── ...
├── libtorch/                   ← LibTorch 2.11.0（arm64）
├── onnxruntime/                ← 需手动下载
└── bin/                        ← 输出可执行文件
```

---

## 💡 推荐工作流

1. **第一次启动**：
   ```bash
   ./build.sh libtorch && ./bin/IndustryVisionKit
   ```

2. **修改代码后**：
   ```bash
   cd build && make -j8 && cd .. && ./bin/IndustryVisionKit
   ```

3. **完全重新构建**：
   ```bash
   ./build.sh clean && ./build.sh libtorch
   ```

---

## 📞 更多信息

- 📖 详细编译指南：查看 `BUILD_GUIDE.md`
- 🤖 YOLO 模型：支持 v5, v7, v8, v9, v11 等，放在 `resource/models/`
- 📊 检测结果：可导出为 CSV 格式

祝使用愉快！🎉
