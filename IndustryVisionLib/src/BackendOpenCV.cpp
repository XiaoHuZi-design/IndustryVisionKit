#include "IndustryVisionLib/InferenceBackend.h"

#include <QString>
#include <QFileInfo>

#include <opencv2/dnn.hpp>

#include <cstring>
#include <memory>
#include <vector>

namespace IndustryVision {

class BackendOpenCV : public InferenceBackend {
public:
    QString name() const override { return QStringLiteral("OpenCV DNN"); }

    bool loadModel(const QString& modelPath,
                   int& inputWidth, int& inputHeight,
                   QString* message) override {
        try {
            m_net = cv::dnn::readNetFromONNX(modelPath.toStdString());
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

            // OpenCV DNN 无法直接读取静态 shape，使用默认 640x640
            inputWidth = 640;
            inputHeight = 640;

            m_loaded = true;
            if (message) {
                *message = QStringLiteral("模型已加载：%1，推理后端：OpenCV DNN")
                               .arg(QFileInfo(modelPath).fileName());
            }
            return true;
        } catch (const cv::Exception& e) {
            if (message) {
                *message = QStringLiteral("OpenCV DNN 加载失败：%1").arg(QString::fromStdString(e.what()));
            }
            return false;
        }
    }

    bool isReady() const override {
        return m_loaded && !m_net.empty();
    }

    bool infer(const float* inputData, int inputWidth, int inputHeight,
               std::vector<float>& outputData,
               std::vector<int64_t>& outputShape) override {
        if (!isReady()) return false;

        // 数据已经预处理好了（NCHW float32），直接构造 blob
        const int shape[] = {1, 3, inputHeight, inputWidth};
        const cv::Mat blob(4, shape, CV_32F, const_cast<float*>(inputData));

        m_net.setInput(blob);

        // forward 到第一个输出层
        cv::Mat out = m_net.forward();
        if (out.empty()) return false;

        // 输出 shape
        outputShape.clear();
        for (int i = 0; i < out.size.dims(); ++i) {
            outputShape.push_back(out.size[i]);
        }

        // 拷贝输出数据
        const size_t total = static_cast<size_t>(out.total());
        outputData.resize(total);
        std::memcpy(outputData.data(), out.ptr<float>(), total * sizeof(float));

        return true;
    }

private:
    cv::dnn::Net m_net;
    bool m_loaded = false;
};

// 工厂函数
std::unique_ptr<InferenceBackend> createOpenCVBackend() {
    return std::make_unique<BackendOpenCV>();
}

} // namespace IndustryVision
