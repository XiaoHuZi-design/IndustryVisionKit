#pragma once

#include <QDateTime>
#include <QImage>
#include <QList>
#include <QRect>
#include <QString>

namespace IndustryVision {

enum class InputSourceMode {
    Image,
    Video,
    Camera
};

struct DetectionResult {
    QString className;
    double confidence = 0.0;
    QRect boundingBox;
};

struct DetectionConfig {
    QString modelPath;
    QString classFilePath;
    QString modelVersion;
    QString backendName;  // "ONNX Runtime" / "OpenVINO" / "LibTorch"，空则自动选择
    double confidenceThreshold = 0.25;
    double iouThreshold = 0.45;
};

struct DetectionReport {
    QList<DetectionResult> results;
    QString summaryText;
    QString sourceName;
    QDateTime timestamp;
    QImage annotatedImage;
};

} // namespace IndustryVision
