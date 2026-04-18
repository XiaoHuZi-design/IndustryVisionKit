#include "IndustryVisionLib/InferenceBackend.h"

#ifdef INDUSTRYVISION_HAS_OPENVINO

#include <QString>
#include <QFileInfo>

#include <openvino/openvino.hpp>

#include <cstring>
#include <memory>
#include <vector>

namespace IndustryVision {

class BackendOpenVINO : public InferenceBackend {
public:
    QString name() const override { return QStringLiteral("OpenVINO"); }

    bool loadModel(const QString& modelPath,
                   int& inputWidth, int& inputHeight,
                   QString* message) override {
        try {
            const QFileInfo fileInfo(modelPath);

            std::shared_ptr<ov::Model> model = m_core.read_model(
                fileInfo.absoluteFilePath().toStdString());

            m_compiledModel = m_core.compile_model(model, "AUTO");
            m_inferRequest = m_compiledModel.create_infer_request();

            // 从模型获取输入尺寸
            const ov::PartialShape inputPartialShape = model->input().get_partial_shape();
            if (inputPartialShape.rank().get_length() == 4) {
                inputHeight = inputPartialShape[2].is_static()
                    ? static_cast<int>(inputPartialShape[2].get_length()) : 640;
                inputWidth = inputPartialShape[3].is_static()
                    ? static_cast<int>(inputPartialShape[3].get_length()) : 640;
            } else {
                inputWidth = 640;
                inputHeight = 640;
            }

            m_initialized = true;
            if (message) {
                *message = QStringLiteral("模型已加载：%1，推理后端：OpenVINO (AUTO)")
                               .arg(QFileInfo(modelPath).fileName());
            }
            return true;
        } catch (const std::exception& e) {
            if (message) {
                *message = QStringLiteral("OpenVINO 加载失败：%1").arg(QString::fromStdString(e.what()));
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
            ov::Tensor inputTensor(ov::element::f32,
                {1, 3, static_cast<unsigned long>(inputHeight), static_cast<unsigned long>(inputWidth)});
            std::memcpy(inputTensor.data<float>(), inputData,
                        static_cast<size_t>(3 * inputWidth * inputHeight) * sizeof(float));
            m_inferRequest.set_input_tensor(inputTensor);
            m_inferRequest.infer();

            ov::Tensor outputTensor = m_inferRequest.get_output_tensor();
            const float* rawOutput = outputTensor.data<float>();
            const ov::Shape ovShape = outputTensor.get_shape();

            outputShape.clear();
            for (auto d : ovShape) {
                outputShape.push_back(static_cast<int64_t>(d));
            }

            const size_t total = outputTensor.get_size();
            outputData.resize(total);
            std::memcpy(outputData.data(), rawOutput, total * sizeof(float));

            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

private:
    ov::Core m_core;
    ov::CompiledModel m_compiledModel;
    ov::InferRequest m_inferRequest;
    bool m_initialized = false;
};

std::unique_ptr<InferenceBackend> createOpenVINOBackend() {
    return std::make_unique<BackendOpenVINO>();
}

} // namespace IndustryVision

#endif // INDUSTRYVISION_HAS_OPENVINO
