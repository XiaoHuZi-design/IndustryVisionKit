#pragma once

#include "IndustryVisionLib/YoloEngine.h"

#include <QWidget>
#include <opencv2/videoio.hpp>

QT_BEGIN_NAMESPACE
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QRadioButton;
class QSpinBox;
class QTableWidget;
class QTextEdit;
class QTimer;
QT_END_NAMESPACE

namespace IndustryVision::Gui {

class DetectionWidget final : public QWidget {
    Q_OBJECT

public:
    explicit DetectionWidget(QWidget* parent = nullptr);

    void setCurrentUser(const QString& username);

signals:
    void logoutRequested();
    void statusMessageChanged(const QString& message);

private slots:
    void browseModel();
    void loadModel();
    void applyParameters();
    void selectInputSource();
    void runDetection();
    void stopDetection();
    void clearResults();
    void exportResults();
    void clearLogs();
    void processStreamingFrame();

private:
    void buildUi();
    void applyStyle();
    void appendLog(const QString& text);
    void updateStatusDisplay(const QString& detectionStatus);
    void updateImagePreview(QLabel* label, const QImage& image);
    void populateTable(const QList<IndustryVision::DetectionResult>& results);
    QImage createPlaceholderFrame(const QString& title, const QString& subtitle, int frameIndex) const;
    void performDetection(const QImage& image, const QString& sourceName);
    void closeCapture();
    void refreshCameras();
    IndustryVision::InputSourceMode currentMode() const;

    IndustryVision::YoloEngine m_engine;
    IndustryVision::DetectionConfig m_config;
    QString m_currentUser;
    QString m_currentSourcePath;
    QList<IndustryVision::DetectionResult> m_lastResults;
    QImage m_lastOriginalImage;
    QImage m_lastAnnotatedImage;
    QTimer* m_streamTimer = nullptr;
    int m_frameCounter = 0;
    cv::VideoCapture m_videoCapture;
    bool m_captureOpen = false;

    QLineEdit* m_modelPathEdit = nullptr;
    QLineEdit* m_classFileEdit = nullptr;
    QComboBox* m_versionCombo = nullptr;
    QComboBox* m_backendCombo = nullptr;
    QDoubleSpinBox* m_confidenceSpin = nullptr;
    QDoubleSpinBox* m_iouSpin = nullptr;
    QRadioButton* m_imageModeButton = nullptr;
    QRadioButton* m_videoModeButton = nullptr;
    QRadioButton* m_cameraModeButton = nullptr;
    QPushButton* m_selectSourceButton = nullptr;
    QComboBox* m_cameraCombo = nullptr;
    QPushButton* m_refreshCameraButton = nullptr;
    QWidget* m_cameraRow = nullptr;
    QLabel* m_modelStatusValue = nullptr;
    QLabel* m_detectionStatusValue = nullptr;
    QLabel* m_currentSourceValue = nullptr;
    QProgressBar* m_progressBar = nullptr;
    QLabel* m_originalImageLabel = nullptr;
    QLabel* m_resultImageLabel = nullptr;
    QTableWidget* m_resultTable = nullptr;
    QTextEdit* m_summaryText = nullptr;
    QPlainTextEdit* m_logEdit = nullptr;
    QLabel* m_currentUserLabel = nullptr;
};

} // namespace IndustryVision::Gui
