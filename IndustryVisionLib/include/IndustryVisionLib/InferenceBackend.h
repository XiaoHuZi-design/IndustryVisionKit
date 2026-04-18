#pragma once

#include <QString>
#include <QImage>
#include <QList>
#include <vector>
#include <memory>

namespace IndustryVision {

struct DetectionResult;
struct DetectionConfig;

// 共享预处理信息，各后端统一使用
struct PreprocessInfo {
    std::vector<float> tensor;  // NCHW float32，已归一化
    int inputWidth = 640;
    int inputHeight = 640;
    float scale = 1.0f;
    int padX = 0;
    int padY = 0;
};

// 推理后端抽象接口，每种推理引擎实现一个子类
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    // 后端名称，如 "OpenCV DNN", "ONNX Runtime", "OpenVINO", "LibTorch"
    virtual QString name() const = 0;

    // 加载模型，成功返回 true。通过 inputWidth/inputHeight 返回模型期望的输入尺寸
    virtual bool loadModel(const QString& modelPath,
                           int& inputWidth, int& inputHeight,
                           QString* message) = 0;

    // 模型是否已加载可推理
    virtual bool isReady() const = 0;

    // 执行推理：输入已预处理的 NCHW tensor，输出原始 float 数据和 shape
    // 返回 true 表示推理成功，outputData 和 outputShape 被填充
    virtual bool infer(const float* inputData, int inputWidth, int inputHeight,
                       std::vector<float>& outputData,
                       std::vector<int64_t>& outputShape) = 0;
};

} // namespace IndustryVision
