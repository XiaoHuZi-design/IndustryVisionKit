#include "IndustryVisionLib/InferenceBackend.h"

#ifdef INDUSTRYVISION_HAS_LIBTORCH

// Qt 的 slots/signals 宏会和 LibTorch 头文件冲突，必须在 include torch 之前取消定义
#undef slots
#undef signals

#include <torch/torch.h>
#include <torch/script.h>

#include <QString>
#include <QFileInfo>

#include <cstring>
#include <memory>

namespace IndustryVision {

class BackendLibTorch : public InferenceBackend {
public:
    QString name() const override { return QStringLiteral("LibTorch"); }

    bool loadModel(const QString& modelPath,
                   int& inputWidth, int& inputHeight,
                   QString* message) override {
        try {
            m_module = torch::jit::load(modelPath.toStdString());
            m_module.to(m_device);
            m_module.eval();

            // 单线程推理，避免 OpenMP 和 TBB/OpenCV 在同一进程冲突
            torch::set_num_threads(1);

            // LibTorch TorchScript 无法直接读取静态 shape，使用默认 640x640
            inputWidth = 640;
            inputHeight = 640;

            m_initialized = true;
            if (message) {
                *message = QStringLiteral("模型已加载：%1，推理后端：LibTorch (CPU)")
                               .arg(QFileInfo(modelPath).fileName());
            }
            return true;
        } catch (const std::exception& e) {
            if (message) {
                *message = QStringLiteral("LibTorch 加载失败：%1").arg(QString::fromStdString(e.what()));
            }
            return false;
        }
    }

    bool isReady() const override {
        return m_initialized;
    }

    bool infer(const float* inputData, int inputWidth, int inputHeight,
               std::vector<float>& outputData,
               std::vector<int64_t>& outputShape) override {
        if (!isReady()) return false;

        try {
            torch::NoGradGuard noGrad;
            auto inputTensor = torch::from_blob(
                const_cast<float*>(inputData),
                {1, 3, inputHeight, inputWidth},
                torch::kFloat32).clone().to(m_device);

            auto output = m_module.forward({inputTensor});

            // 有些模型输出是 tuple/list，需要取第一个元素
            if (!output.isTensor()) {
                if (output.isTuple()) {
                    output = output.toTuple()->elements()[0];
                } else if (output.isList()) {
                    output = output.toList().get(0);
                }
            }
            if (!output.isTensor()) return false;

            auto outputTensor = output.toTensor().contiguous().to(torch::kCPU);
            const int64_t numel = outputTensor.numel();
            outputData.resize(static_cast<size_t>(numel));
            std::memcpy(outputData.data(), outputTensor.data_ptr<float>(),
                        static_cast<size_t>(numel) * sizeof(float));

            for (auto s : outputTensor.sizes()) {
                outputShape.push_back(s);
            }

            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

private:
    torch::jit::script::Module m_module;
    torch::Device m_device{torch::kCPU};
    bool m_initialized = false;
};

std::unique_ptr<InferenceBackend> createLibTorchBackend() {
    return std::make_unique<BackendLibTorch>();
}

} // namespace IndustryVision

#endif // INDUSTRYVISION_HAS_LIBTORCH
