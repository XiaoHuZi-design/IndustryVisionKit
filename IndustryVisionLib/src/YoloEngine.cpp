#include "IndustryVisionLib/YoloEngine.h"
#include "IndustryVisionLib/InferenceBackend.h"

#include <QDir>
#include <QColor>
#include <QFileInfo>
#include <QFile>
#include <QMap>
#include <QPainter>
#include <QPen>
#include <QRandomGenerator>
#include <QRegularExpression>
#include <QStringList>
#include <QTextStream>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace IndustryVision {

// ===================== 工厂函数声明（各 Backend*.cpp 提供）=====================

std::unique_ptr<InferenceBackend> createOpenCVBackend();

#ifdef INDUSTRYVISION_HAS_ONNXRUNTIME
std::unique_ptr<InferenceBackend> createOnnxRuntimeBackend();
#endif

#ifdef INDUSTRYVISION_HAS_OPENVINO
std::unique_ptr<InferenceBackend> createOpenVINOBackend();
#endif

#ifdef INDUSTRYVISION_HAS_LIBTORCH
std::unique_ptr<InferenceBackend> createLibTorchBackend();
#endif

// ===================== 匿名辅助函数 =====================

namespace {

struct CandidateBox {
    int classIndex = -1;
    float score = 0.0F;
    QRectF rect;
};

QStringList readLines(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }

    QStringList lines;
    QTextStream stream(&file);
    while (!stream.atEnd()) {
        const QString line = stream.readLine().trimmed();
        if (!line.isEmpty()) {
            lines.append(line);
        }
    }
    return lines;
}

float intersectionOverUnion(const QRectF& lhs, const QRectF& rhs) {
    const QRectF intersection = lhs.intersected(rhs);
    const float intersectionArea = static_cast<float>(intersection.width() * intersection.height());
    if (intersectionArea <= 0.0F) {
        return 0.0F;
    }

    const float unionArea = static_cast<float>(lhs.width() * lhs.height() + rhs.width() * rhs.height()) - intersectionArea;
    return unionArea > 0.0F ? intersectionArea / unionArea : 0.0F;
}

QList<DetectionResult> applyNms(const std::vector<CandidateBox>& candidates,
                                const QStringList& classNames,
                                const QSize& imageSize,
                                double iouThreshold) {
    std::vector<int> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&candidates](int lhs, int rhs) {
        return candidates.at(lhs).score > candidates.at(rhs).score;
    });

    std::vector<bool> removed(candidates.size(), false);
    QList<DetectionResult> results;

    for (int orderIndex = 0; orderIndex < static_cast<int>(order.size()); ++orderIndex) {
        const int candidateIndex = order.at(orderIndex);
        if (removed.at(candidateIndex)) {
            continue;
        }

        const CandidateBox& candidate = candidates.at(candidateIndex);
        DetectionResult result;
        result.className = (candidate.classIndex >= 0 && candidate.classIndex < classNames.size())
                               ? classNames.at(candidate.classIndex)
                               : QStringLiteral("class_%1").arg(candidate.classIndex);
        result.confidence = candidate.score;

        QRect rect = candidate.rect.toAlignedRect();
        rect = rect.intersected(QRect(QPoint(0, 0), imageSize));
        result.boundingBox = rect;
        results.append(result);

        for (int compareIndex = orderIndex + 1; compareIndex < static_cast<int>(order.size()); ++compareIndex) {
            const int nextIndex = order.at(compareIndex);
            if (removed.at(nextIndex)) {
                continue;
            }

            const CandidateBox& nextCandidate = candidates.at(nextIndex);
            if (candidate.classIndex != nextCandidate.classIndex) {
                continue;
            }

            if (intersectionOverUnion(candidate.rect, nextCandidate.rect) > static_cast<float>(iouThreshold)) {
                removed[nextIndex] = true;
            }
        }
    }

    return results;
}

// 共享预处理：letterbox + HWC→CHW + /255.0
PreprocessInfo preprocessImage(const QImage& image, int inputWidth, int inputHeight) {
    const QImage rgbImage = image.convertToFormat(QImage::Format_RGB888);

    const float scale = std::min(static_cast<float>(inputWidth) / static_cast<float>(rgbImage.width()),
                                 static_cast<float>(inputHeight) / static_cast<float>(rgbImage.height()));
    const int resizedWidth = std::max(1, static_cast<int>(std::round(rgbImage.width() * scale)));
    const int resizedHeight = std::max(1, static_cast<int>(std::round(rgbImage.height() * scale)));
    const int padX = (inputWidth - resizedWidth) / 2;
    const int padY = (inputHeight - resizedHeight) / 2;

    QImage letterboxed(inputWidth, inputHeight, QImage::Format_RGB888);
    letterboxed.fill(QColor(114, 114, 114));

    QPainter painter(&letterboxed);
    painter.drawImage(QRect(padX, padY, resizedWidth, resizedHeight), rgbImage);
    painter.end();

    std::vector<float> tensor(static_cast<size_t>(3 * inputWidth * inputHeight));
    for (int y = 0; y < inputHeight; ++y) {
        const uchar* row = letterboxed.constScanLine(y);
        for (int x = 0; x < inputWidth; ++x) {
            const int pixelOffset = x * 3;
            const size_t baseIndex = static_cast<size_t>(y * inputWidth + x);
            tensor[baseIndex] = static_cast<float>(row[pixelOffset]) / 255.0F;
            tensor[static_cast<size_t>(inputWidth * inputHeight) + baseIndex] =
                static_cast<float>(row[pixelOffset + 1]) / 255.0F;
            tensor[static_cast<size_t>(2 * inputWidth * inputHeight) + baseIndex] =
                static_cast<float>(row[pixelOffset + 2]) / 255.0F;
        }
    }

    return {std::move(tensor), inputWidth, inputHeight, scale, padX, padY};
}

// 共享后处理：解析 YOLO 输出 float 数组
QList<DetectionResult> parseYoloOutput(const float* outputData,
                                       const std::vector<int64_t>& shape,
                                       const PreprocessInfo& info,
                                       const QImage& originalImage,
                                       const DetectionConfig& config,
                                       const QStringList& classNames) {
    if (shape.size() != 3 || outputData == nullptr) {
        return {};
    }

    const int64_t dim1 = shape.at(1);
    const int64_t dim2 = shape.at(2);
    std::vector<CandidateBox> candidates;

    // YOLOv26 End-to-End 格式：(1, 300, 6) → [x1,y1,x2,y2, score, class_id]
    if (config.modelVersion == QStringLiteral("YOLOv26") && dim2 == 6) {
        const int64_t detCount = dim1;
        candidates.reserve(static_cast<size_t>(detCount));
        for (int64_t i = 0; i < detCount; ++i) {
            const float x1 = outputData[i * 6 + 0];
            const float y1 = outputData[i * 6 + 1];
            const float x2 = outputData[i * 6 + 2];
            const float y2 = outputData[i * 6 + 3];
            const float score = outputData[i * 6 + 4];
            const int classIndex = static_cast<int>(std::round(outputData[i * 6 + 5]));

            if (score < static_cast<float>(config.confidenceThreshold)) {
                continue;
            }

            const float origX1 = (x1 - static_cast<float>(info.padX)) / info.scale;
            const float origY1 = (y1 - static_cast<float>(info.padY)) / info.scale;
            const float origX2 = (x2 - static_cast<float>(info.padX)) / info.scale;
            const float origY2 = (y2 - static_cast<float>(info.padY)) / info.scale;

            CandidateBox candidate;
            candidate.classIndex = classIndex;
            candidate.score = score;
            candidate.rect = QRectF(QPointF(std::max(0.0F, origX1), std::max(0.0F, origY1)),
                                    QPointF(std::min(static_cast<float>(originalImage.width() - 1), origX2),
                                            std::min(static_cast<float>(originalImage.height() - 1), origY2)));
            if (candidate.rect.width() > 1.0 && candidate.rect.height() > 1.0) {
                candidates.push_back(candidate);
            }
        }
        return applyNms(candidates, classNames, originalImage.size(), config.iouThreshold);
    }

    // --- 标准 YOLO 格式解析（v5/v8/v11）---
    bool transposed = false;
    int64_t boxCount = 0;
    int64_t featureCount = 0;

    if (dim2 >= 6 && dim2 <= 512) {
        boxCount = dim1;
        featureCount = dim2;
    } else {
        boxCount = dim2;
        featureCount = dim1;
        transposed = true;
    }

    const bool hasObjectness = config.modelVersion == QStringLiteral("YOLOv5");
    const int classCount = static_cast<int>(hasObjectness ? (featureCount - 5) : (featureCount - 4));
    if (classCount <= 0) {
        return {};
    }

    candidates.reserve(static_cast<size_t>(boxCount));

    const auto valueAt = [outputData, boxCount, featureCount, transposed](int64_t row, int64_t col) -> float {
        return transposed ? outputData[col * boxCount + row] : outputData[row * featureCount + col];
    };

    for (int64_t row = 0; row < boxCount; ++row) {
        const float centerX = valueAt(row, 0);
        const float centerY = valueAt(row, 1);
        const float width = valueAt(row, 2);
        const float height = valueAt(row, 3);
        const float objectness = hasObjectness ? valueAt(row, 4) : 1.0F;

        int bestClass = -1;
        float bestScore = 0.0F;
        for (int classIndex = 0; classIndex < classCount; ++classIndex) {
            const float classScore = valueAt(row, (hasObjectness ? 5 : 4) + classIndex);
            const float score = classScore * objectness;
            if (score > bestScore) {
                bestScore = score;
                bestClass = classIndex;
            }
        }

        if (bestScore < static_cast<float>(config.confidenceThreshold)) {
            continue;
        }

        const float x1 = (centerX - width / 2.0F - static_cast<float>(info.padX)) / info.scale;
        const float y1 = (centerY - height / 2.0F - static_cast<float>(info.padY)) / info.scale;
        const float x2 = (centerX + width / 2.0F - static_cast<float>(info.padX)) / info.scale;
        const float y2 = (centerY + height / 2.0F - static_cast<float>(info.padY)) / info.scale;

        CandidateBox candidate;
        candidate.classIndex = bestClass;
        candidate.score = bestScore;
        candidate.rect = QRectF(QPointF(std::max(0.0F, x1), std::max(0.0F, y1)),
                                QPointF(std::min(static_cast<float>(originalImage.width() - 1), x2),
                                        std::min(static_cast<float>(originalImage.height() - 1), y2)));
        if (candidate.rect.width() > 1.0 && candidate.rect.height() > 1.0) {
            candidates.push_back(candidate);
        }
    }

    return applyNms(candidates, classNames, originalImage.size(), config.iouThreshold);
}

} // namespace

// ===================== YoloEngine =====================

YoloEngine::YoloEngine(QObject* parent)
    : QObject(parent) {
}

YoloEngine::~YoloEngine() = default;

QStringList YoloEngine::supportedVersions() const {
    return {
        QStringLiteral("YOLOv5"),
        QStringLiteral("YOLOv8"),
        QStringLiteral("YOLOv11"),
        QStringLiteral("YOLOv26"),
    };
}

QStringList YoloEngine::availableBackends() const {
    QStringList backends;
    // OpenCV DNN 始终可用（OpenCV 是必须依赖）
    backends.append(QStringLiteral("OpenCV DNN"));
#ifdef INDUSTRYVISION_HAS_ONNXRUNTIME
    backends.append(QStringLiteral("ONNX Runtime"));
#endif
#ifdef INDUSTRYVISION_HAS_OPENVINO
    backends.append(QStringLiteral("OpenVINO"));
#endif
#ifdef INDUSTRYVISION_HAS_LIBTORCH
    backends.append(QStringLiteral("LibTorch"));
#endif
    return backends;
}

bool YoloEngine::loadModel(const DetectionConfig& config, QString* message) {
    const QFileInfo fileInfo(config.modelPath);
    if (!fileInfo.exists() || !fileInfo.isFile()) {
        if (message != nullptr) {
            *message = QStringLiteral("模型文件不存在，请重新选择。");
        }
        m_modelLoaded = false;
        return false;
    }

    if (!supportedVersions().contains(config.modelVersion)) {
        if (message != nullptr) {
            *message = QStringLiteral("不支持的 YOLO 版本。");
        }
        m_modelLoaded = false;
        return false;
    }

    m_config = config;
    m_classNames = loadClassNames(config.modelPath);

    // 选择后端
    QString backend = config.backendName;
    if (backend.isEmpty()) {
        backend = availableBackends().first();  // 默认 OpenCV DNN
    }

    // 释放旧后端
    m_backend.reset();

    // 创建对应后端
    if (backend == QStringLiteral("OpenCV DNN")) {
        m_backend = createOpenCVBackend();
    }
#ifdef INDUSTRYVISION_HAS_ONNXRUNTIME
    else if (backend == QStringLiteral("ONNX Runtime")) {
        m_backend = createOnnxRuntimeBackend();
    }
#endif
#ifdef INDUSTRYVISION_HAS_OPENVINO
    else if (backend == QStringLiteral("OpenVINO")) {
        m_backend = createOpenVINOBackend();
    }
#endif
#ifdef INDUSTRYVISION_HAS_LIBTORCH
    else if (backend == QStringLiteral("LibTorch")) {
        m_backend = createLibTorchBackend();
    }
#endif

    if (!m_backend) {
        m_modelLoaded = true;
        if (message != nullptr) {
            *message = QStringLiteral("%1 已加载：%2，当前为模拟模式（后端 %3 不可用）。")
                           .arg(config.modelVersion, fileInfo.fileName(), backend);
        }
        return true;
    }

    int w = 640, h = 640;
    if (m_backend->loadModel(config.modelPath, w, h, message)) {
        m_inputWidth = w;
        m_inputHeight = h;
        m_modelLoaded = true;
        return true;
    }

    // 后端加载失败，回退模拟模式
    m_backend.reset();
    m_modelLoaded = true;
    if (message != nullptr) {
        *message = QStringLiteral("%1 已加载：%2，后端加载失败，已切换为模拟模式。")
                       .arg(config.modelVersion, fileInfo.fileName());
    }
    return true;
}

void YoloEngine::setThresholds(double confidenceThreshold, double iouThreshold) {
    m_config.confidenceThreshold = confidenceThreshold;
    m_config.iouThreshold = iouThreshold;
}

bool YoloEngine::isModelLoaded() const {
    return m_modelLoaded;
}

bool YoloEngine::isRealBackendActive() const {
    return m_backend && m_backend->isReady();
}

QString YoloEngine::backendName() const {
    if (m_backend && m_backend->isReady()) {
        return m_backend->name();
    }
    return QStringLiteral("Simulator");
}

DetectionReport YoloEngine::detect(const QImage& image, const QString& sourceName) const {
    DetectionReport report;
    report.sourceName = sourceName;
    report.timestamp = QDateTime::currentDateTime();

    if (!m_modelLoaded || image.isNull()) {
        report.summaryText = QStringLiteral("检测失败：模型未加载或输入源为空。");
        return report;
    }

    report.results = isRealBackendActive() ? runRealInference(image) : generateResults(image.size(), sourceName);
    report.annotatedImage = drawResults(image, report.results);
    report.summaryText = buildSummary(report.results) + QStringLiteral(" 当前后端：%1。").arg(backendName());
    return report;
}

QStringList YoloEngine::loadClassNames(const QString& modelPath) const {
    const QFileInfo modelFile(modelPath);
    const QDir classesDir(modelFile.absoluteDir().absoluteFilePath(QStringLiteral("../classes")));

    if (!m_config.classFilePath.isEmpty()) {
        const QStringList lines = readLines(m_config.classFilePath);
        if (!lines.isEmpty()) {
            return lines;
        }
    }

    const QString defaultPath = classesDir.absoluteFilePath(QStringLiteral("coco.names.txt"));
    const QStringList lines = readLines(defaultPath);
    if (!lines.isEmpty()) {
        return lines;
    }

    return {};
}

// ===================== 推理（统一入口）=====================

QList<DetectionResult> YoloEngine::runRealInference(const QImage& image) const {
    if (!m_backend || !m_backend->isReady() || image.isNull()) return {};

    // 共享预处理
    const PreprocessInfo info = preprocessImage(image, m_inputWidth, m_inputHeight);

    // 后端推理
    std::vector<float> outputData;
    std::vector<int64_t> outputShape;
    if (!m_backend->infer(info.tensor.data(), m_inputWidth, m_inputHeight, outputData, outputShape)) {
        return {};
    }

    // 共享后处理
    return parseYoloOutput(outputData.data(), outputShape, info, image, m_config, m_classNames);
}

// ===================== 模拟模式 =====================

QList<DetectionResult> YoloEngine::generateResults(const QSize& imageSize, const QString& sourceName) const {
    QList<DetectionResult> results;
    if (imageSize.isEmpty()) {
        return results;
    }

    const quint32 seed = qHash(sourceName + m_config.modelPath + m_config.modelVersion + QString::number(imageSize.width()));
    QRandomGenerator generator(seed);
    const int count = 2 + static_cast<int>(seed % 5);

    for (int index = 0; index < count; ++index) {
        const int width = qMax(80, imageSize.width() / (4 + generator.bounded(3)));
        const int height = qMax(80, imageSize.height() / (4 + generator.bounded(3)));
        const int left = generator.bounded(qMax(1, imageSize.width() - width));
        const int top = generator.bounded(qMax(1, imageSize.height() - height));
        const double confidence = 0.45 + generator.generateDouble() * 0.5;

        if (confidence < m_config.confidenceThreshold) {
            continue;
        }

        DetectionResult result;
        result.className = m_classNames.isEmpty()
                               ? QStringLiteral("object")
                               : m_classNames.at(generator.bounded(m_classNames.size()));
        result.confidence = confidence;
        result.boundingBox = QRect(left, top, width, height);
        results.append(result);
    }

    return results;
}

// ===================== 绘制结果 =====================

QImage YoloEngine::drawResults(const QImage& image, const QList<DetectionResult>& results) const {
    QImage canvas = image.convertToFormat(QImage::Format_ARGB32_Premultiplied);
    QPainter painter(&canvas);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QList<QColor> palette = {
        QColor(231, 76, 60),
        QColor(46, 204, 113),
        QColor(52, 152, 219),
        QColor(241, 196, 15),
        QColor(155, 89, 182),
        QColor(230, 126, 34),
        QColor(26, 188, 156),
        QColor(236, 100, 165),
        QColor(22, 160, 133),
        QColor(192, 57, 43),
        QColor(41, 128, 185),
        QColor(142, 68, 173),
        QColor(39, 174, 96),
        QColor(211, 84, 0),
        QColor(44, 62, 80),
    };

    QMap<QString, int> classColorMap;
    int nextColorIndex = 0;

    const int fontSize = qBound(10, qMin(canvas.width(), canvas.height()) / 40, 22);
    QFont font(painter.font());
    font.setPixelSize(fontSize);
    font.setBold(true);
    painter.setFont(font);

    const QFontMetrics fm(font);
    const int labelHeight = fontSize + 6;
    const int penWidth = qBound(2, fontSize / 5, 4);
    const int labelPadX = 6;

    for (int index = 0; index < results.size(); ++index) {
        const DetectionResult& result = results.at(index);

        if (!classColorMap.contains(result.className)) {
            classColorMap[result.className] = nextColorIndex % palette.size();
            ++nextColorIndex;
        }
        const QColor color = palette.at(classColorMap.value(result.className));

        painter.setPen(QPen(color, penWidth));
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(result.boundingBox);

        const QString caption = QStringLiteral("%1 %2")
                                    .arg(result.className)
                                    .arg(result.confidence, 0, 'f', 2);

        const int textWidth = fm.horizontalAdvance(caption);
        const int labelWidth = textWidth + labelPadX * 2;

        const QRect labelRect(
            result.boundingBox.left(),
            qMax(0, result.boundingBox.top() - labelHeight - 2),
            qMin(labelWidth, result.boundingBox.width()),
            labelHeight);

        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 200));
        painter.drawRoundedRect(labelRect, 3, 3);

        painter.setPen(Qt::white);
        painter.drawText(labelRect.adjusted(labelPadX, 0, -labelPadX, 0),
                         Qt::AlignVCenter | Qt::AlignLeft, caption);
    }

    painter.end();
    return canvas;
}

QString YoloEngine::buildSummary(const QList<DetectionResult>& results) const {
    if (results.isEmpty()) {
        return QStringLiteral("检测结果汇总：未检测到目标。");
    }

    QMap<QString, int> counter;
    for (const DetectionResult& result : results) {
        counter[result.className] += 1;
    }

    QStringList fragments;
    for (auto it = counter.cbegin(); it != counter.cend(); ++it) {
        fragments.append(QStringLiteral("%1: %2个").arg(it.key()).arg(it.value()));
    }

    return QStringLiteral("检测结果汇总：共检测到 %1 个目标，%2。")
        .arg(results.size())
        .arg(fragments.join(QStringLiteral("，")));
}

} // namespace IndustryVision
