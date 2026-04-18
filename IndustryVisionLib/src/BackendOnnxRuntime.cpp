#include "IndustryVisionLib/InferenceBackend.h"

#ifdef INDUSTRYVISION_HAS_ONNXRUNTIME

#include <QString>
#include <QFileInfo>

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstring>
#include <memory>

namespace IndustryVision {

class BackendOnnxRuntime : public InferenceBackend {
public:
    QString name() const override { return QStringLiteral("ONNX Runtime"); }

    bool loadModel(const QString& modelPath,
                   int& inputWidth, int& inputHeight,
                   QString* message) override {
        try {
            m_sessionOptions.SetIntraOpNumThreads(1);
            m_sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            const QFileInfo fileInfo(modelPath);
            const std::string path = fileInfo.absoluteFilePath().toStdString();
            m_session = std::make_unique<Ort::Session>(m_env, path.c_str(), m_sessionOptions);

            // 获取输入输出名称
            Ort::AllocatorWithDefaultOptions allocator;
            const auto inputName = m_session->GetInputNameAllocated(0, allocator);
            const auto outputName = m_session->GetOutputNameAllocated(0, allocator);
            m_inputName = QString::fromUtf8(inputName.get());
            m_outputName = QString::fromUtf8(outputName.get());

            // 从模型获取输入尺寸
            const Ort::TypeInfo inputTypeInfo = m_session->GetInputTypeInfo(0);
            const auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
            const std::vector<int64_t> inputShape = inputTensorInfo.GetShape();
            if (inputShape.size() == 4) {
                inputHeight = inputShape.at(2) > 0 ? static_cast<int>(inputShape.at(2)) : 640;
                inputWidth = inputShape.at(3) > 0 ? static_cast<int>(inputShape.at(3)) : 640;
            } else {
                inputWidth = 640;
                inputHeight = 640;
            }

            m_loaded = true;
            if (message) {
                *message = QStringLiteral("模型已加载：%1，推理后端：ONNX Runtime")
                               .arg(QFileInfo(modelPath).fileName());
            }
            return true;
        } catch (const std::exception& e) {
            if (message) {
                *message = QStringLiteral("ONNX Runtime 加载失败：%1").arg(QString::fromUtf8(e.what()));
            }
            return false;
        }
    }

    bool isReady() const override {
        return m_loaded && m_session != nullptr;
    }

    bool infer(const float* inputData, int inputWidth, int inputHeight,
               std::vector<float>& outputData,
               std::vector<int64_t>& outputShape) override {
        if (!isReady()) return false;

        try {
            const std::array<int64_t, 4> inputShapeArr = {1, 3, inputHeight, inputWidth};
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputValue = Ort::Value::CreateTensor<float>(
                memoryInfo,
                const_cast<float*>(inputData),
                static_cast<size_t>(3 * inputWidth * inputHeight),
                inputShapeArr.data(),
                inputShapeArr.size());

            const QByteArray inputNameUtf8 = m_inputName.toUtf8();
            const QByteArray outputNameUtf8 = m_outputName.toUtf8();
            const std::array<const char*, 1> inputNames = {inputNameUtf8.constData()};
            const std::array<const char*, 1> outputNames = {outputNameUtf8.constData()};

            auto outputValues = m_session->Run(
                Ort::RunOptions{nullptr},
                inputNames.data(),
                &inputValue,
                1,
                outputNames.data(),
                1);

            if (outputValues.empty()) return false;

            const auto& outputValue = outputValues.front();
            const auto tensorInfo = outputValue.GetTensorTypeAndShapeInfo();
            outputShape = tensorInfo.GetShape();
            const float* rawOutput = outputValue.GetTensorData<float>();

            const size_t total = tensorInfo.GetElementCount();
            outputData.resize(total);
            std::memcpy(outputData.data(), rawOutput, total * sizeof(float));

            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

private:
    Ort::Env m_env{ORT_LOGGING_LEVEL_WARNING, "IndustryVisionKit"};
    Ort::SessionOptions m_sessionOptions;
    std::unique_ptr<Ort::Session> m_session;
    QString m_inputName;
    QString m_outputName;
    bool m_loaded = false;
};

std::unique_ptr<InferenceBackend> createOnnxRuntimeBackend() {
    return std::make_unique<BackendOnnxRuntime>();
}

} // namespace IndustryVision

#endif // INDUSTRYVISION_HAS_ONNXRUNTIME
