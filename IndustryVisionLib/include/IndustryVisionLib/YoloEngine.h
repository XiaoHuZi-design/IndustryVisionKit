#pragma once

#include "IndustryVisionLib/DetectionTypes.h"

#include <memory>
#include <QObject>
#include <QStringList>

namespace IndustryVision {

struct RuntimeState;

class YoloEngine final : public QObject {
    Q_OBJECT

public:
    explicit YoloEngine(QObject* parent = nullptr);
    ~YoloEngine() override;

    QStringList supportedVersions() const;
    bool loadModel(const DetectionConfig& config, QString* message);
    void setThresholds(double confidenceThreshold, double iouThreshold);

    [[nodiscard]] bool isModelLoaded() const;
    [[nodiscard]] bool isRealBackendActive() const;
    [[nodiscard]] QString backendName() const;
    DetectionReport detect(const QImage& image, const QString& sourceName) const;

private:
    QStringList loadClassNames(const QString& modelPath) const;
    QList<DetectionResult> runRealInference(const QImage& image) const;
    QList<DetectionResult> generateResults(const QSize& imageSize, const QString& sourceName) const;
    QImage drawResults(const QImage& image, const QList<DetectionResult>& results) const;
    QString buildSummary(const QList<DetectionResult>& results) const;

    DetectionConfig m_config;
    QStringList m_classNames;
    std::unique_ptr<RuntimeState> m_runtimeState;
    bool m_modelLoaded = false;
};

} // namespace IndustryVision
