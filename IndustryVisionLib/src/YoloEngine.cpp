#include "IndustryVisionLib/YoloEngine.h"

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
#include <numeric>
#include <vector>

#ifdef INDUSTRYVISION_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#ifdef INDUSTRYVISION_HAS_LIBTORCH
#include <torch/torch.h>
#endif

namespace IndustryVision {

struct RuntimeState {
#ifdef INDUSTRYVISION_HAS_ONNXRUNTIME
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
    QString inputName;
    QString outputName;
    int inputWidth = 640;
    int inputHeight = 640;
    bool layoutNchw = true;

    RuntimeState()
        : env(ORT_LOGGING_LEVEL_WARNING, "IndustryVisionKit") {
    }
#endif

#ifdef INDUSTRYVISION_HAS_LIBTORCH
    torch::jit::script::Module module;
    torch::Device device{torch::kCPU};
    int inputWidth = 640;
    int inputHeight = 640;
    bool initialized = false;
#endif

#if !defined(INDUSTRYVISION_HAS_ONNXRUNTIME) && !defined(INDUSTRYVISION_HAS_LIBTORCH)
    RuntimeState() { }
#endif
};

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

} // namespace

YoloEngine::YoloEngine(QObject* parent)
    : QObject(parent)
    , m_runtimeState(std::make_unique<RuntimeState>()) {
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

bool YoloEngine::loadModel(const DetectionConfig& config, QString* message) {
    // 统一检查模型版本与文件可用性，便于未来替换成真实推理后端。
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
    m_modelLoaded = true;

#ifdef INDUSTRYVISION_HAS_LIBTORCH
    try {
        m_runtimeState = std::make_unique<RuntimeState>();
        
        // LibTorch JIT 模型加载（需要导出为 traced 或 scripted 模型）
        m_runtimeState->module = torch::jit::load(modelPath);
        m_runtimeState->module.to(m_runtimeState->device);
        m_runtimeState->module.eval();
        m_runtimeState->initialized = true;

        if (message != nullptr) {
            *message = QStringLiteral("%1 已加载：%2，推理后端：LibTorch (CPU)")
                           .arg(config.modelVersion, QFileInfo(config.modelPath).fileName());
        }
        return true;
    } catch (const std::exception& exception) {
        if (message != nullptr) {
            *message = QStringLiteral("%1 已加载：%2，LibTorch 初始化失败，已切换为模拟模式。原因：%3")
                           .arg(config.modelVersion, QFileInfo(config.modelPath).fileName(), QString::fromUtf8(exception.what()));
        }
        return true;
    }
#elif defined(INDUSTRYVISION_HAS_ONNXRUNTIME)
    try {
        m_runtimeState = std::make_unique<RuntimeState>();
        m_runtimeState->sessionOptions.SetIntraOpNumThreads(1);
        m_runtimeState->sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        const std::string modelPath = QFileInfo(config.modelPath).absoluteFilePath().toStdString();
        m_runtimeState->session = std::make_unique<Ort::Session>(
            m_runtimeState->env,
            modelPath.c_str(),
            m_runtimeState->sessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;
        const auto inputName = m_runtimeState->session->GetInputNameAllocated(0, allocator);
        const auto outputName = m_runtimeState->session->GetOutputNameAllocated(0, allocator);
        m_runtimeState->inputName = QString::fromUtf8(inputName.get());
        m_runtimeState->outputName = QString::fromUtf8(outputName.get());

        const Ort::TypeInfo inputTypeInfo = m_runtimeState->session->GetInputTypeInfo(0);
        const auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> inputShape = inputTensorInfo.GetShape();
        if (inputShape.size() == 4) {
            m_runtimeState->inputHeight = inputShape.at(2) > 0 ? static_cast<int>(inputShape.at(2)) : 640;
            m_runtimeState->inputWidth = inputShape.at(3) > 0 ? static_cast<int>(inputShape.at(3)) : 640;
        }

        if (message != nullptr) {
            *message = QStringLiteral("%1 已加载：%2，推理后端：ONNX Runtime")
                           .arg(config.modelVersion, fileInfo.fileName());
        }
        return true;
    } catch (const std::exception& exception) {
        if (message != nullptr) {
            *message = QStringLiteral("%1 已加载：%2，ONNX Runtime 初始化失败，已切换为模拟模式。原因：%3")
                           .arg(config.modelVersion, fileInfo.fileName(), QString::fromUtf8(exception.what()));
        }
        return true;
    }
#else
    if (message != nullptr) {
        *message = QStringLiteral("%1 已加载：%2，当前为模拟模式。下载并配置 LibTorch 或 ONNX Runtime 后可启用真实推理。")
                       .arg(config.modelVersion, fileInfo.fileName());
    }
    return true;
#endif
}

void YoloEngine::setThresholds(double confidenceThreshold, double iouThreshold) {
    m_config.confidenceThreshold = confidenceThreshold;
    m_config.iouThreshold = iouThreshold;
}

bool YoloEngine::isModelLoaded() const {
    return m_modelLoaded;
}

bool YoloEngine::isRealBackendActive() const {
#ifdef INDUSTRYVISION_HAS_LIBTORCH
    return m_runtimeState != nullptr && m_runtimeState->initialized;
#elif defined(INDUSTRYVISION_HAS_ONNXRUNTIME)
    return m_runtimeState != nullptr && m_runtimeState->session != nullptr;
#else
    return false;
#endif
}

QString YoloEngine::backendName() const {
#ifdef INDUSTRYVISION_HAS_LIBTORCH
    if (m_runtimeState != nullptr && m_runtimeState->initialized) {
        return QStringLiteral("LibTorch (CPU)");
    }
#endif
    if (isRealBackendActive()) {
        return QStringLiteral("ONNX Runtime");
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

    // 优先使用用户显式指定的类别文件
    if (!m_config.classFilePath.isEmpty()) {
        const QStringList lines = readLines(m_config.classFilePath);
        if (!lines.isEmpty()) {
            return lines;
        }
    }

    // 默认加载 coco.names.txt
    const QString defaultPath = classesDir.absoluteFilePath(QStringLiteral("coco.names.txt"));
    const QStringList lines = readLines(defaultPath);
    if (!lines.isEmpty()) {
        return lines;
    }

    return {};
}

QList<DetectionResult> YoloEngine::runRealInference(const QImage& image) const {
#ifdef INDUSTRYVISION_HAS_ONNXRUNTIME
    if (m_runtimeState == nullptr || m_runtimeState->session == nullptr || image.isNull()) {
        return {};
    }

    const int inputWidth = m_runtimeState->inputWidth;
    const int inputHeight = m_runtimeState->inputHeight;
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

    std::vector<float> inputTensor(static_cast<size_t>(3 * inputWidth * inputHeight));
    for (int y = 0; y < inputHeight; ++y) {
        const uchar* row = letterboxed.constScanLine(y);
        for (int x = 0; x < inputWidth; ++x) {
            const int pixelOffset = x * 3;
            const size_t baseIndex = static_cast<size_t>(y * inputWidth + x);
            inputTensor[baseIndex] = static_cast<float>(row[pixelOffset]) / 255.0F;
            inputTensor[static_cast<size_t>(inputWidth * inputHeight) + baseIndex] =
                static_cast<float>(row[pixelOffset + 1]) / 255.0F;
            inputTensor[static_cast<size_t>(2 * inputWidth * inputHeight) + baseIndex] =
                static_cast<float>(row[pixelOffset + 2]) / 255.0F;
        }
    }

    const std::array<int64_t, 4> inputShape = {1, 3, inputHeight, inputWidth};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputValue = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensor.data(),
        inputTensor.size(),
        inputShape.data(),
        inputShape.size());

    const QByteArray inputNameUtf8 = m_runtimeState->inputName.toUtf8();
    const QByteArray outputNameUtf8 = m_runtimeState->outputName.toUtf8();
    const std::array<const char*, 1> inputNames = {inputNameUtf8.constData()};
    const std::array<const char*, 1> outputNames = {outputNameUtf8.constData()};
    auto outputValues = m_runtimeState->session->Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        &inputValue,
        1,
        outputNames.data(),
        1);

    if (outputValues.empty()) {
        return {};
    }

    const auto& outputValue = outputValues.front();
    const auto tensorInfo = outputValue.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = tensorInfo.GetShape();
    const float* outputData = outputValue.GetTensorData<float>();
    const size_t elementCount = tensorInfo.GetElementCount();
    if (shape.size() != 3 || elementCount == 0) {
        return {};
    }

    const int64_t dim1 = shape.at(1);
    const int64_t dim2 = shape.at(2);
    std::vector<CandidateBox> candidates;

    // YOLOv26 End-to-End 格式：(1, 300, 6) → [x1,y1,x2,y2, score, class_id]
    // 直接输出最终检测结果，无需 NMS
    if (m_config.modelVersion == QStringLiteral("YOLOv26") && dim2 == 6) {
        const int64_t detCount = dim1;
        candidates.reserve(static_cast<size_t>(detCount));
        for (int64_t i = 0; i < detCount; ++i) {
            const float x1 = outputData[i * 6 + 0];
            const float y1 = outputData[i * 6 + 1];
            const float x2 = outputData[i * 6 + 2];
            const float y2 = outputData[i * 6 + 3];
            const float score = outputData[i * 6 + 4];
            const int classIndex = static_cast<int>(std::round(outputData[i * 6 + 5]));

            if (score < static_cast<float>(m_config.confidenceThreshold)) {
                continue;
            }

            // 坐标从 letterboxed 空间还原到原图
            const float origX1 = (x1 - static_cast<float>(padX)) / scale;
            const float origY1 = (y1 - static_cast<float>(padY)) / scale;
            const float origX2 = (x2 - static_cast<float>(padX)) / scale;
            const float origY2 = (y2 - static_cast<float>(padY)) / scale;

            CandidateBox candidate;
            candidate.classIndex = classIndex;
            candidate.score = score;
            candidate.rect = QRectF(QPointF(std::max(0.0F, origX1), std::max(0.0F, origY1)),
                                    QPointF(std::min(static_cast<float>(image.width() - 1), origX2),
                                            std::min(static_cast<float>(image.height() - 1), origY2)));
            if (candidate.rect.width() > 1.0 && candidate.rect.height() > 1.0) {
                candidates.push_back(candidate);
            }
        }
        // E2E 模型已做过 NMS，直接返回
        return applyNms(candidates, m_classNames, image.size(), m_config.iouThreshold);
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

    // YOLOv5 输出含 objectness 置信度列：[x,y,w,h,obj, class1,class2,...]
    // YOLOv8/v11 输出不含 objectness：[x,y,w,h, class1,class2,...]
    const bool hasObjectness = m_config.modelVersion == QStringLiteral("YOLOv5");
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

        if (bestScore < static_cast<float>(m_config.confidenceThreshold)) {
            continue;
        }

        const float x1 = (centerX - width / 2.0F - static_cast<float>(padX)) / scale;
        const float y1 = (centerY - height / 2.0F - static_cast<float>(padY)) / scale;
        const float x2 = (centerX + width / 2.0F - static_cast<float>(padX)) / scale;
        const float y2 = (centerY + height / 2.0F - static_cast<float>(padY)) / scale;

        CandidateBox candidate;
        candidate.classIndex = bestClass;
        candidate.score = bestScore;
        candidate.rect = QRectF(QPointF(std::max(0.0F, x1), std::max(0.0F, y1)),
                                QPointF(std::min(static_cast<float>(image.width() - 1), x2),
                                        std::min(static_cast<float>(image.height() - 1), y2)));
        if (candidate.rect.width() > 1.0 && candidate.rect.height() > 1.0) {
            candidates.push_back(candidate);
        }
    }

    return applyNms(candidates, m_classNames, image.size(), m_config.iouThreshold);
#else
    Q_UNUSED(image);
    return {};
#endif
}

QList<DetectionResult> YoloEngine::generateResults(const QSize& imageSize, const QString& sourceName) const {
    QList<DetectionResult> results;
    if (imageSize.isEmpty()) {
        return results;
    }

    // 使用稳定种子生成可复现结果，方便 UI 联调与流程验证。
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

QImage YoloEngine::drawResults(const QImage& image, const QList<DetectionResult>& results) const {
    QImage canvas = image.convertToFormat(QImage::Format_ARGB32_Premultiplied);
    QPainter painter(&canvas);
    painter.setRenderHint(QPainter::Antialiasing, true);

    // Per-class color palette — consistent hash so same class always same color
    const QList<QColor> palette = {
        QColor(231, 76, 60),    // red
        QColor(46, 204, 113),   // green
        QColor(52, 152, 219),   // blue
        QColor(241, 196, 15),   // yellow
        QColor(155, 89, 182),   // purple
        QColor(230, 126, 34),   // orange
        QColor(26, 188, 156),   // teal
        QColor(236, 100, 165),  // pink
        QColor(22, 160, 133),   // dark teal
        QColor(192, 57, 43),    // dark red
        QColor(41, 128, 185),   // dark blue
        QColor(142, 68, 173),   // dark purple
        QColor(39, 174, 96),    // dark green
        QColor(211, 84, 0),     // dark orange
        QColor(44, 62, 80),     // navy
    };

    // Map className → stable color index
    QMap<QString, int> classColorMap;
    int nextColorIndex = 0;

    // Font size proportional to image dimensions
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

        // Assign stable color per class
        if (!classColorMap.contains(result.className)) {
            classColorMap[result.className] = nextColorIndex % palette.size();
            ++nextColorIndex;
        }
        const QColor color = palette.at(classColorMap.value(result.className));

        // Draw bounding box
        painter.setPen(QPen(color, penWidth));
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(result.boundingBox);

        // Caption text
        const QString caption = QStringLiteral("%1 %2")
                                    .arg(result.className)
                                    .arg(result.confidence, 0, 'f', 2);

        // Measure text width for label rect
        const int textWidth = fm.horizontalAdvance(caption);
        const int labelWidth = textWidth + labelPadX * 2;

        const QRect labelRect(
            result.boundingBox.left(),
            qMax(0, result.boundingBox.top() - labelHeight - 2),
            qMin(labelWidth, result.boundingBox.width()),
            labelHeight);

        // Semi-transparent filled rounded rect
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 200));
        painter.drawRoundedRect(labelRect, 3, 3);

        // White text
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
