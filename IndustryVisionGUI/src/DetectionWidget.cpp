#include "IndustryVisionGUI/DetectionWidget.h"

#include <QAbstractItemView>
#include <QCoreApplication>
#include <QComboBox>
#include <QDateTime>
#include <QDoubleSpinBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QImageReader>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPlainTextEdit>

#include <QProgressBar>
#include <QPushButton>
#include <QPixmap>
#include <QRadioButton>
#include <QScrollArea>
#include <QTabWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTextEdit>
#include <QTextStream>
#include <QTimer>
#include <QVBoxLayout>

#include <opencv2/imgproc.hpp>

namespace IndustryVision::Gui {

namespace {

QString withTimestamp(const QString& text) {
    return QStringLiteral("[%1] %2")
        .arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd hh:mm:ss")))
        .arg(text);
}

QString projectResourcePath(const QString& relativePath) {
    const QString candidate = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("../") + relativePath);
    if (QFileInfo::exists(candidate)) {
        return QFileInfo(candidate).absoluteFilePath();
    }
    return QString();
}

QString defaultModelForVersion(const QString& version) {
    static const QHash<QString, QString> defaults = {
        {QStringLiteral("YOLOv5"), QStringLiteral("resource/models/yolov5s.onnx")},
        {QStringLiteral("YOLOv8"), QStringLiteral("resource/models/yolov8n.onnx")},
        {QStringLiteral("YOLOv11"), QStringLiteral("resource/models/yolo11n.onnx")},
        {QStringLiteral("YOLOv26"), QStringLiteral("resource/models/yolo26n.onnx")},
    };
    const QString relPath = defaults.value(version, QString());
    return relPath.isEmpty() ? QString() : projectResourcePath(relPath);
}

} // namespace

DetectionWidget::DetectionWidget(QWidget* parent)
    : QWidget(parent)
    , m_streamTimer(new QTimer(this)) {
    buildUi();
    applyStyle();

    m_versionCombo->addItems(m_engine.supportedVersions());
    m_config.modelVersion = m_versionCombo->currentText();
    m_modelPathEdit->setText(defaultModelForVersion(m_versionCombo->currentText()));
    m_currentSourcePath = projectResourcePath(QStringLiteral("resource/images/bus.jpg"));
    if (!m_currentSourcePath.isEmpty()) {
        m_currentSourceValue->setText(QFileInfo(m_currentSourcePath).fileName());
    }
    appendLog(QStringLiteral("默认模型：%1").arg(m_modelPathEdit->text().isEmpty() ? QStringLiteral("未找到") : m_modelPathEdit->text()));
    appendLog(QStringLiteral("默认图片：%1").arg(m_currentSourcePath.isEmpty() ? QStringLiteral("未找到") : m_currentSourcePath));

    connect(m_streamTimer, &QTimer::timeout, this, &DetectionWidget::processStreamingFrame);
    m_streamTimer->setInterval(650);
}

void DetectionWidget::setCurrentUser(const QString& username) {
    m_currentUser = username;
    if (m_currentUserLabel) {
        m_currentUserLabel->setText(username);
    }
    appendLog(QStringLiteral("当前登录用户：%1").arg(username));
    emit statusMessageChanged(QStringLiteral("当前用户：%1").arg(username));
}

void DetectionWidget::browseModel() {
    const QString path = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("选择模型文件"),
        QString(),
        QStringLiteral("Model Files (*.pt *.onnx *.engine *.bin);;All Files (*)"));

    if (!path.isEmpty()) {
        m_modelPathEdit->setText(path);
        appendLog(QStringLiteral("已选择模型文件：%1").arg(path));
    }
}

void DetectionWidget::loadModel() {
    m_config.modelPath = m_modelPathEdit->text().trimmed();
    m_config.classFilePath = m_classFileEdit->text().trimmed();
    m_config.modelVersion = m_versionCombo->currentText();
    m_config.confidenceThreshold = m_confidenceSpin->value();
    m_config.iouThreshold = m_iouSpin->value();

    QString message;
    if (m_engine.loadModel(m_config, &message)) {
        m_modelStatusValue->setText(QStringLiteral("已加载"));
        appendLog(message);
        appendLog(QStringLiteral("当前后端：%1").arg(m_engine.backendName()));
        updateStatusDisplay(QStringLiteral("待检测"));
        m_progressBar->setValue(100);
    } else {
        m_modelStatusValue->setText(QStringLiteral("未加载"));
        appendLog(message);
        updateStatusDisplay(QStringLiteral("加载失败"));
        m_progressBar->setValue(0);
    }
}

void DetectionWidget::applyParameters() {
    m_engine.setThresholds(m_confidenceSpin->value(), m_iouSpin->value());
    appendLog(QStringLiteral("参数已应用：置信度 %1，IOU %2")
                  .arg(m_confidenceSpin->value(), 0, 'f', 2)
                  .arg(m_iouSpin->value(), 0, 'f', 2));
    emit statusMessageChanged(QStringLiteral("检测参数已更新"));
}

void DetectionWidget::refreshCameras() {
    m_cameraCombo->clear();

    // 用 OpenCV 实际探测可打开的设备，索引和分辨率都是真实的
    for (int i = 0; i < 8; ++i) {
        cv::VideoCapture probe(i);
        if (probe.isOpened()) {
            const int w = static_cast<int>(probe.get(cv::CAP_PROP_FRAME_WIDTH));
            const int h = static_cast<int>(probe.get(cv::CAP_PROP_FRAME_HEIGHT));
            const QString label = QStringLiteral("摄像头 %1 (%2x%3)").arg(i).arg(w).arg(h);
            m_cameraCombo->addItem(label, i);
            probe.release();
        }
    }

    if (m_cameraCombo->count() == 0) {
        m_cameraCombo->addItem(QStringLiteral("未检测到摄像头"), -1);
    }
    appendLog(QStringLiteral("摄像头刷新完成，检测到 %1 个设备").arg(
        m_cameraCombo->count() == 1 && m_cameraCombo->currentData().toInt() < 0 ? 0 : m_cameraCombo->count()));
}

void DetectionWidget::selectInputSource() {
    if (currentMode() == IndustryVision::InputSourceMode::Camera) {
        const int camIndex = m_cameraCombo->currentData().toInt();
        if (camIndex < 0) {
            appendLog(QStringLiteral("未检测到可用摄像头。"));
            return;
        }
        m_currentSourcePath = QStringLiteral("摄像头 %1").arg(camIndex);
        m_currentSourceValue->setText(m_currentSourcePath);
        appendLog(QStringLiteral("已选择：%1").arg(m_currentSourcePath));
        return;
    }

    const bool imageMode = currentMode() == IndustryVision::InputSourceMode::Image;
    const QString filter = imageMode
                               ? QStringLiteral("Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
                               : QStringLiteral("Videos (*.mp4 *.avi *.mov *.mkv);;All Files (*)");
    const QString path = QFileDialog::getOpenFileName(
        this,
        imageMode ? QStringLiteral("选择图片") : QStringLiteral("选择视频"),
        QString(),
        filter);

    if (!path.isEmpty()) {
        m_currentSourcePath = path;
        m_currentSourceValue->setText(QFileInfo(path).fileName());
        appendLog(QStringLiteral("已选择输入源：%1").arg(path));
    }
}

void DetectionWidget::runDetection() {
    if (!m_engine.isModelLoaded()) {
        appendLog(QStringLiteral("执行失败：请先加载模型。"));
        updateStatusDisplay(QStringLiteral("模型未加载"));
        return;
    }

    m_progressBar->setValue(15);
    updateStatusDisplay(QStringLiteral("检测中"));

    if (currentMode() == IndustryVision::InputSourceMode::Image) {
        if (m_currentSourcePath.isEmpty()) {
            appendLog(QStringLiteral("图片模式下请先选择图片。"));
            updateStatusDisplay(QStringLiteral("等待图片"));
            return;
        }

        QImageReader reader(m_currentSourcePath);
        reader.setAutoTransform(true);
        const QImage image = reader.read();
        performDetection(image, QFileInfo(m_currentSourcePath).fileName());
        return;
    }

    // 视频/摄像头模式：打开 VideoCapture
    closeCapture();

    if (currentMode() == IndustryVision::InputSourceMode::Video) {
        if (m_currentSourcePath.isEmpty()) {
            appendLog(QStringLiteral("视频模式下请先选择视频文件。"));
            updateStatusDisplay(QStringLiteral("等待视频"));
            return;
        }
        m_videoCapture.open(m_currentSourcePath.toStdString());
    } else {
        // 摄像头：使用选择的设备索引
        const int camIndex = m_cameraCombo->currentData().toInt();
        m_videoCapture.open(camIndex >= 0 ? camIndex : 0);
    }

    if (!m_videoCapture.isOpened()) {
        appendLog(QStringLiteral("无法打开输入源：%1").arg(m_currentSourcePath));
        updateStatusDisplay(QStringLiteral("打开失败"));
        m_progressBar->setValue(0);
        return;
    }

    m_captureOpen = true;
    m_frameCounter = 0;
    m_streamTimer->start();
    appendLog(currentMode() == IndustryVision::InputSourceMode::Video
                  ? QStringLiteral("开始视频流检测：%1").arg(m_currentSourcePath)
                  : QStringLiteral("开始摄像头流检测。"));
}

void DetectionWidget::stopDetection() {
    m_streamTimer->stop();
    closeCapture();
    updateStatusDisplay(QStringLiteral("已停止"));
    appendLog(QStringLiteral("检测流程已停止。"));
    emit statusMessageChanged(QStringLiteral("检测已停止"));
}

void DetectionWidget::clearResults() {
    m_streamTimer->stop();
    closeCapture();
    m_resultTable->setRowCount(0);
    m_summaryText->clear();
    m_logEdit->clear();
    m_lastResults.clear();
    m_lastOriginalImage = QImage();
    m_lastAnnotatedImage = QImage();
    m_originalImageLabel->setText(QStringLiteral("原始图像预览"));
    m_resultImageLabel->setText(QStringLiteral("检测结果预览"));
    m_progressBar->setValue(0);
    m_currentSourceValue->setText(QStringLiteral("无"));
    m_currentSourcePath.clear();
    updateStatusDisplay(QStringLiteral("空闲"));
}

void DetectionWidget::exportResults() {
    if (m_lastResults.isEmpty()) {
        appendLog(QStringLiteral("暂无可导出的检测结果。"));
        return;
    }

    const QString path = QFileDialog::getSaveFileName(
        this,
        QStringLiteral("导出检测结果"),
        QStringLiteral("detection_results.csv"),
        QStringLiteral("CSV Files (*.csv);;Text Files (*.txt)"));

    if (path.isEmpty()) {
        return;
    }

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        appendLog(QStringLiteral("检测结果导出失败。"));
        return;
    }

    QTextStream stream(&file);
    const bool csvFormat = path.endsWith(QStringLiteral(".csv"), Qt::CaseInsensitive);
    const QString separator = csvFormat ? QStringLiteral(",") : QStringLiteral(" | ");
    stream << QStringList{QStringLiteral("类型"), QStringLiteral("置信度"), QStringLiteral("位置"), QStringLiteral("大小")}.join(separator) << '\n';

    for (const IndustryVision::DetectionResult& result : m_lastResults) {
        stream << result.className << separator
               << QString::number(result.confidence, 'f', 2) << separator
               << QStringLiteral("(%1,%2)").arg(result.boundingBox.x()).arg(result.boundingBox.y()) << separator
               << QStringLiteral("%1x%2").arg(result.boundingBox.width()).arg(result.boundingBox.height())
               << '\n';
    }

    file.close();
    appendLog(QStringLiteral("检测结果已导出：%1").arg(path));
}

void DetectionWidget::clearLogs() {
    m_logEdit->clear();
    appendLog(QStringLiteral("日志已清空。"));
}

void DetectionWidget::closeCapture() {
    if (m_captureOpen) {
        m_videoCapture.release();
        m_captureOpen = false;
    }
}

void DetectionWidget::processStreamingFrame() {
    if (!m_captureOpen || !m_videoCapture.isOpened()) {
        m_streamTimer->stop();
        appendLog(QStringLiteral("输入源已断开，停止检测。"));
        updateStatusDisplay(QStringLiteral("源已断开"));
        closeCapture();
        return;
    }

    cv::Mat frame;
    if (!m_videoCapture.read(frame) || frame.empty()) {
        // 视频文件播放完毕
        if (currentMode() == IndustryVision::InputSourceMode::Video) {
            appendLog(QStringLiteral("视频播放完毕，共 %1 帧。").arg(m_frameCounter));
            m_streamTimer->stop();
            closeCapture();
            updateStatusDisplay(QStringLiteral("视频结束"));
        } else {
            appendLog(QStringLiteral("摄像头读取失败。"));
            m_streamTimer->stop();
            closeCapture();
            updateStatusDisplay(QStringLiteral("读取失败"));
        }
        return;
    }

    ++m_frameCounter;

    // cv::Mat (BGR) → QImage (RGB)
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    const QImage qimage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                        QImage::Format_RGB888);

    const QString sourceName = currentMode() == IndustryVision::InputSourceMode::Video
                                   ? QStringLiteral("%1 - frame %2").arg(QFileInfo(m_currentSourcePath).fileName()).arg(m_frameCounter)
                                   : QStringLiteral("Camera-0 - frame %1").arg(m_frameCounter);

    // QImage 引用 cv::Mat 数据，必须拷贝后使用
    performDetection(qimage.copy(), sourceName);
}

void DetectionWidget::buildUi() {
    // 整体布局：左侧控制面板(固定宽) + 右侧预览区(自适应) / 下方结果+日志
    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(8, 8, 8, 8);
    rootLayout->setSpacing(8);
    setLayout(rootLayout);

    // ========== 上方区域 ==========
    auto* topLayout = new QHBoxLayout();
    topLayout->setSpacing(8);
    topLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->addLayout(topLayout, 7);

    // ----- 左侧控制面板 -----
    auto* controlPanel = new QWidget(this);
    controlPanel->setObjectName(QStringLiteral("controlPanel"));
    controlPanel->setFixedWidth(300);

    auto* panelScroll = new QScrollArea(this);
    panelScroll->setWidget(controlPanel);
    panelScroll->setWidgetResizable(true);
    panelScroll->setFixedWidth(318);
    panelScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto* panelLayout = new QVBoxLayout(controlPanel);
    panelLayout->setContentsMargins(10, 10, 10, 10);
    panelLayout->setSpacing(8);

    // --- 用户信息栏 ---
    auto* userBar = new QWidget(controlPanel);
    userBar->setObjectName(QStringLiteral("userBar"));
    auto* userBarLayout = new QHBoxLayout(userBar);
    userBarLayout->setContentsMargins(10, 8, 10, 8);
    userBarLayout->setSpacing(8);

    auto* userIcon = new QLabel(QStringLiteral("\342\230\273"), userBar);  // ☻
    userIcon->setObjectName(QStringLiteral("userIcon"));
    userIcon->setFixedSize(30, 30);
    userIcon->setAlignment(Qt::AlignCenter);

    auto* userVBox = new QVBoxLayout();
    userVBox->setSpacing(0);
    auto* userTitleLabel = new QLabel(QStringLiteral("当前用户"), userBar);
    userTitleLabel->setObjectName(QStringLiteral("userTitleLabel"));
    auto* userNameLabel = new QLabel(QStringLiteral("--"), userBar);
    userNameLabel->setObjectName(QStringLiteral("userNameLabel"));
    m_currentUserLabel = userNameLabel;
    userVBox->addWidget(userTitleLabel);
    userVBox->addWidget(userNameLabel);

    userBarLayout->addWidget(userIcon);
    userBarLayout->addLayout(userVBox, 1);
    panelLayout->addWidget(userBar);

    // --- 模型控制 ---
    auto* modelGroup = new QGroupBox(QStringLiteral("模型控制"), controlPanel);
    auto* modelLayout = new QVBoxLayout(modelGroup);
    modelLayout->setContentsMargins(10, 14, 10, 8);
    modelLayout->setSpacing(5);

    m_versionCombo = new QComboBox(modelGroup);
    auto* versionLabel = new QLabel(QStringLiteral("YOLO 版本:"), modelGroup);
    versionLabel->setObjectName(QStringLiteral("fieldLabel"));

    m_modelPathEdit = new QLineEdit(modelGroup);
    m_modelPathEdit->setPlaceholderText(QStringLiteral("选择模型文件路径..."));
    auto* browseButton = new QPushButton(QStringLiteral("浏览"), modelGroup);
    browseButton->setObjectName(QStringLiteral("secondaryButton"));
    browseButton->setFixedWidth(50);

    auto* modelRow = new QHBoxLayout();
    modelRow->setSpacing(4);
    modelRow->addWidget(m_modelPathEdit, 1);
    modelRow->addWidget(browseButton, 0);

    auto* classLabel = new QLabel(QStringLiteral("类别文件 (可选):"), modelGroup);
    classLabel->setObjectName(QStringLiteral("fieldLabel"));

    m_classFileEdit = new QLineEdit(modelGroup);
    m_classFileEdit->setPlaceholderText(QStringLiteral("默认 coco.names.txt"));
    auto* browseClassButton = new QPushButton(QStringLiteral("浏览"), modelGroup);
    browseClassButton->setObjectName(QStringLiteral("secondaryButton"));
    browseClassButton->setFixedWidth(50);

    auto* classRow = new QHBoxLayout();
    classRow->setSpacing(4);
    classRow->addWidget(m_classFileEdit, 1);
    classRow->addWidget(browseClassButton, 0);

    modelLayout->addWidget(versionLabel);
    modelLayout->addWidget(m_versionCombo);
    modelLayout->addLayout(modelRow);
    modelLayout->addWidget(classLabel);
    modelLayout->addLayout(classRow);
    panelLayout->addWidget(modelGroup);

    // --- 参数控制 ---
    auto* parameterGroup = new QGroupBox(QStringLiteral("参数控制"), controlPanel);
    auto* parameterLayout = new QGridLayout(parameterGroup);
    parameterLayout->setContentsMargins(14, 14, 10, 8);
    parameterLayout->setSpacing(5);

    m_confidenceSpin = new QDoubleSpinBox(parameterGroup);
    m_confidenceSpin->setRange(0.01, 1.00);
    m_confidenceSpin->setSingleStep(0.05);
    m_confidenceSpin->setValue(0.25);
    m_confidenceSpin->setDecimals(2);

    m_iouSpin = new QDoubleSpinBox(parameterGroup);
    m_iouSpin->setRange(0.01, 1.00);
    m_iouSpin->setSingleStep(0.05);
    m_iouSpin->setValue(0.45);
    m_iouSpin->setDecimals(2);

    auto* confLabel = new QLabel(QStringLiteral("置信度:"), parameterGroup);
    confLabel->setObjectName(QStringLiteral("fieldLabel"));
    auto* iouLabel = new QLabel(QStringLiteral("IOU 阈值:"), parameterGroup);
    iouLabel->setObjectName(QStringLiteral("fieldLabel"));

    parameterLayout->addWidget(confLabel, 0, 0);
    parameterLayout->addWidget(m_confidenceSpin, 0, 1);
    parameterLayout->addWidget(iouLabel, 1, 0);
    parameterLayout->addWidget(m_iouSpin, 1, 1);

    panelLayout->addWidget(parameterGroup);

    // --- 输入源 ---
    auto* sourceGroup = new QGroupBox(QStringLiteral("输入源"), controlPanel);
    auto* sourceLayout = new QVBoxLayout(sourceGroup);
    sourceLayout->setContentsMargins(10, 14, 10, 8);
    sourceLayout->setSpacing(4);

    auto* modeRow = new QHBoxLayout();
    modeRow->setSpacing(10);
    m_imageModeButton = new QRadioButton(QStringLiteral("图片"), sourceGroup);
    m_videoModeButton = new QRadioButton(QStringLiteral("视频"), sourceGroup);
    m_cameraModeButton = new QRadioButton(QStringLiteral("摄像头"), sourceGroup);
    m_imageModeButton->setChecked(true);
    modeRow->addWidget(m_imageModeButton);
    modeRow->addWidget(m_videoModeButton);
    modeRow->addWidget(m_cameraModeButton);
    modeRow->addStretch();

    // 摄像头选择行（仅摄像头模式显示）
    m_cameraRow = new QWidget(sourceGroup);
    auto* cameraRowLayout = new QHBoxLayout(m_cameraRow);
    cameraRowLayout->setContentsMargins(0, 0, 0, 0);
    cameraRowLayout->setSpacing(4);
    m_cameraRow->setVisible(false);

    m_cameraCombo = new QComboBox(m_cameraRow);
    m_cameraCombo->setObjectName(QStringLiteral("cameraCombo"));
    refreshCameras();  // 初始探测

    m_refreshCameraButton = new QPushButton(QStringLiteral("刷新"), m_cameraRow);
    m_refreshCameraButton->setObjectName(QStringLiteral("secondaryButton"));
    m_refreshCameraButton->setFixedWidth(44);

    cameraRowLayout->addWidget(m_cameraCombo, 1);
    cameraRowLayout->addWidget(m_refreshCameraButton);

    // 图片/视频模式的选择按钮（仅非摄像头模式显示）
    m_selectSourceButton = new QPushButton(QStringLiteral("选择输入源"), sourceGroup);
    m_selectSourceButton->setObjectName(QStringLiteral("secondaryButton"));

    sourceLayout->addLayout(modeRow);
    sourceLayout->addWidget(m_cameraRow);
    sourceLayout->addWidget(m_selectSourceButton);
    panelLayout->addWidget(sourceGroup);

    // --- 检测操作 ---
    auto* actionGroup = new QGroupBox(QStringLiteral("检测操作"), controlPanel);
    auto* actionLayout = new QHBoxLayout(actionGroup);
    actionLayout->setContentsMargins(10, 14, 10, 8);
    actionLayout->setSpacing(4);

    auto* runButton = new QPushButton(QStringLiteral("开始"), actionGroup);
    runButton->setObjectName(QStringLiteral("accentButton"));

    auto* stopButton = new QPushButton(QStringLiteral("停止"), actionGroup);
    stopButton->setObjectName(QStringLiteral("dangerButton"));

    auto* clearButton = new QPushButton(QStringLiteral("清除"), actionGroup);
    clearButton->setObjectName(QStringLiteral("secondaryAction"));

    auto* exportBtn = new QPushButton(QStringLiteral("导出"), actionGroup);
    exportBtn->setObjectName(QStringLiteral("secondaryAction"));

    actionLayout->addWidget(runButton);
    actionLayout->addWidget(stopButton);
    actionLayout->addWidget(clearButton);
    actionLayout->addWidget(exportBtn);
    panelLayout->addWidget(actionGroup);

    // --- 状态显示 ---
    auto* statusGroup = new QGroupBox(QStringLiteral("运行状态"), controlPanel);
    auto* statusLayout = new QGridLayout(statusGroup);
    statusLayout->setContentsMargins(10, 14, 10, 8);
    statusLayout->setSpacing(5);

    auto* modelStatusLabel = new QLabel(QStringLiteral("模型:"), statusGroup);
    modelStatusLabel->setObjectName(QStringLiteral("fieldLabel"));
    m_modelStatusValue = new QLabel(QStringLiteral("未加载"), statusGroup);
    m_modelStatusValue->setObjectName(QStringLiteral("statusValue"));

    auto* detectionStatusLabel = new QLabel(QStringLiteral("检测:"), statusGroup);
    detectionStatusLabel->setObjectName(QStringLiteral("fieldLabel"));
    m_detectionStatusValue = new QLabel(QStringLiteral("空闲"), statusGroup);
    m_detectionStatusValue->setObjectName(QStringLiteral("statusValue"));

    auto* sourceLabel = new QLabel(QStringLiteral("输入:"), statusGroup);
    sourceLabel->setObjectName(QStringLiteral("fieldLabel"));
    m_currentSourceValue = new QLabel(QStringLiteral("--"), statusGroup);
    m_currentSourceValue->setObjectName(QStringLiteral("statusValue"));

    m_progressBar = new QProgressBar(statusGroup);
    m_progressBar->setRange(0, 100);
    m_progressBar->setObjectName(QStringLiteral("progressBar"));
    m_progressBar->setTextVisible(true);

    statusLayout->addWidget(modelStatusLabel, 0, 0);
    statusLayout->addWidget(m_modelStatusValue, 0, 1);
    statusLayout->addWidget(detectionStatusLabel, 1, 0);
    statusLayout->addWidget(m_detectionStatusValue, 1, 1);
    statusLayout->addWidget(sourceLabel, 2, 0);
    statusLayout->addWidget(m_currentSourceValue, 2, 1);
    statusLayout->addWidget(m_progressBar, 3, 0, 1, 2);
    panelLayout->addWidget(statusGroup);

    panelLayout->addStretch(1);

    auto* logoutButton = new QPushButton(QStringLiteral("退出登录"), controlPanel);
    logoutButton->setObjectName(QStringLiteral("logoutButton"));
    panelLayout->addWidget(logoutButton);

    topLayout->addWidget(panelScroll, 0);

    // ----- 右侧预览区 -----
    auto* previewTabs = new QTabWidget(this);
    previewTabs->setObjectName(QStringLiteral("previewTabs"));

    auto* originalPage = new QWidget();
    auto* originalLayout = new QVBoxLayout(originalPage);
    originalLayout->setContentsMargins(4, 4, 4, 4);
    m_originalImageLabel = new QLabel(QStringLiteral("原始图像预览"), originalPage);
    m_originalImageLabel->setAlignment(Qt::AlignCenter);
    m_originalImageLabel->setObjectName(QStringLiteral("imageLabel"));
    m_originalImageLabel->setMinimumSize(320, 240);
    m_originalImageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    originalLayout->addWidget(m_originalImageLabel);

    auto* resultPage = new QWidget();
    auto* resultLayout = new QVBoxLayout(resultPage);
    resultLayout->setContentsMargins(4, 4, 4, 4);
    m_resultImageLabel = new QLabel(QStringLiteral("检测结果预览"), resultPage);
    m_resultImageLabel->setAlignment(Qt::AlignCenter);
    m_resultImageLabel->setObjectName(QStringLiteral("imageLabel"));
    m_resultImageLabel->setMinimumSize(320, 240);
    m_resultImageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    resultLayout->addWidget(m_resultImageLabel);

    previewTabs->addTab(originalPage, QStringLiteral("原始图像"));
    previewTabs->addTab(resultPage, QStringLiteral("检测结果"));
    topLayout->addWidget(previewTabs, 1);

    // ========== 下方区域 ==========
    auto* bottomLayout = new QHBoxLayout();
    bottomLayout->setSpacing(8);
    bottomLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->addLayout(bottomLayout, 2);

    // 检测结果表
    auto* resultGroup = new QGroupBox(QStringLiteral("检测结果"));
    auto* resultGroupLayout = new QVBoxLayout(resultGroup);
    resultGroupLayout->setContentsMargins(8, 12, 8, 6);
    resultGroupLayout->setSpacing(4);

    m_resultTable = new QTableWidget(resultGroup);
    m_resultTable->setColumnCount(4);
    m_resultTable->setHorizontalHeaderLabels({QStringLiteral("检测类型"), QStringLiteral("置信度"), QStringLiteral("位置"), QStringLiteral("大小")});
    m_resultTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_resultTable->horizontalHeader()->setFixedHeight(22);
    m_resultTable->verticalHeader()->setVisible(false);
    m_resultTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_resultTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_resultTable->setAlternatingRowColors(true);
    m_resultTable->setFixedHeight(80);  // 表头22 + 两行各24 + 余量10px ≈ 2.2行

    m_summaryText = new QTextEdit(resultGroup);
    m_summaryText->setReadOnly(true);
    m_summaryText->setMaximumHeight(30);
    m_summaryText->setFixedHeight(30);
    m_summaryText->setObjectName(QStringLiteral("summaryText"));

    resultGroupLayout->addWidget(m_resultTable, 1);
    resultGroupLayout->addWidget(m_summaryText, 0);
    bottomLayout->addWidget(resultGroup, 1);

    // 系统日志
    auto* logGroup = new QGroupBox(QStringLiteral("系统日志"));
    auto* logGroupLayout = new QVBoxLayout(logGroup);
    logGroupLayout->setContentsMargins(8, 12, 8, 6);
    logGroupLayout->setSpacing(4);

    m_logEdit = new QPlainTextEdit(logGroup);
    m_logEdit->setReadOnly(true);
    m_logEdit->setObjectName(QStringLiteral("logEdit"));

    auto* clearLogButton = new QPushButton(QStringLiteral("清除日志"), logGroup);
    clearLogButton->setObjectName(QStringLiteral("secondaryAction"));
    clearLogButton->setFixedWidth(80);

    auto* logButtonLayout = new QHBoxLayout();
    logButtonLayout->addStretch();
    logButtonLayout->addWidget(clearLogButton);

    logGroupLayout->addWidget(m_logEdit, 1);
    logGroupLayout->addLayout(logButtonLayout, 0);
    bottomLayout->addWidget(logGroup, 1);

    // ========== 连接信号 ==========
    connect(browseButton, &QPushButton::clicked, this, [this]() {
        browseModel();
        loadModel();
    });
    connect(browseClassButton, &QPushButton::clicked, this, [this]() {
        const QString path = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("选择类别文件"),
            QString(),
            QStringLiteral("Text Files (*.txt *.names *.csv);;All Files (*)"));
        if (!path.isEmpty()) {
            m_classFileEdit->setText(path);
            appendLog(QStringLiteral("已选择类别文件：%1").arg(path));
            loadModel();
        }
    });
    // 模型路径/类别文件编辑后回车自动加载
    connect(m_modelPathEdit, &QLineEdit::editingFinished, this, &DetectionWidget::loadModel);
    connect(m_classFileEdit, &QLineEdit::editingFinished, this, &DetectionWidget::loadModel);
    // 版本切换自动加载
    connect(m_versionCombo, &QComboBox::currentTextChanged, this, [this](const QString& version) {
        const QString defaultModel = defaultModelForVersion(version);
        if (!defaultModel.isEmpty()) {
            m_modelPathEdit->setText(defaultModel);
            m_classFileEdit->clear();
            appendLog(QStringLiteral("切换至 %1，默认模型：%2").arg(version, QFileInfo(defaultModel).fileName()));
        }
        loadModel();
    });
    // 参数实时应用
    connect(m_confidenceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &DetectionWidget::applyParameters);
    connect(m_iouSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &DetectionWidget::applyParameters);
    connect(m_selectSourceButton, &QPushButton::clicked, this, &DetectionWidget::selectInputSource);
    connect(runButton, &QPushButton::clicked, this, &DetectionWidget::runDetection);
    connect(stopButton, &QPushButton::clicked, this, &DetectionWidget::stopDetection);
    connect(clearButton, &QPushButton::clicked, this, &DetectionWidget::clearResults);
    connect(exportBtn, &QPushButton::clicked, this, &DetectionWidget::exportResults);
    connect(logoutButton, &QPushButton::clicked, this, [this]() {
        m_streamTimer->stop();
        emit logoutRequested();
    });
    connect(clearLogButton, &QPushButton::clicked, this, &DetectionWidget::clearLogs);

    connect(m_imageModeButton, &QRadioButton::toggled, this, [this](bool checked) {
        if (checked) {
            m_cameraRow->setVisible(false);
            m_selectSourceButton->setVisible(true);
            appendLog(QStringLiteral("切换为图片模式"));
        }
    });
    connect(m_videoModeButton, &QRadioButton::toggled, this, [this](bool checked) {
        if (checked) {
            m_cameraRow->setVisible(false);
            m_selectSourceButton->setVisible(true);
            appendLog(QStringLiteral("切换为视频模式"));
        }
    });
    connect(m_cameraModeButton, &QRadioButton::toggled, this, [this](bool checked) {
        if (checked) {
            m_cameraRow->setVisible(true);
            m_selectSourceButton->setVisible(false);
            const int camIndex = m_cameraCombo->currentData().toInt();
            m_currentSourcePath = camIndex >= 0 ? QStringLiteral("摄像头 %1").arg(camIndex) : QString();
            m_currentSourceValue->setText(m_currentSourcePath);
            appendLog(QStringLiteral("切换为摄像头模式"));
        }
    });
    connect(m_cameraCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
        const int camIndex = m_cameraCombo->currentData().toInt();
        m_currentSourcePath = camIndex >= 0 ? QStringLiteral("摄像头 %1").arg(camIndex) : QString();
        m_currentSourceValue->setText(m_currentSourcePath);
        appendLog(QStringLiteral("选择：%1").arg(m_cameraCombo->currentText()));
    });
    connect(m_refreshCameraButton, &QPushButton::clicked, this, &DetectionWidget::refreshCameras);
}

void DetectionWidget::applyStyle() {
    setStyleSheet(QStringLiteral(
        // --- 全局基调 ---
        "QWidget { background-color: #edf1f5; color: #2c3e50; font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; }"

        // --- 左侧控制面板 ---
        "#controlPanel { background-color: #ffffff; }"
        "QScrollArea { background-color: transparent; border: none; }"

        // --- 用户信息栏 ---
        "#userBar { background-color: #f0f5fa; border: 1px solid #dce4ed; border-radius: 6px; }"
        "#userIcon { background-color: #4a90c4; color: white; border-radius: 14px; font-size: 14px; font-weight: bold; }"
        "#userTitleLabel { color: #8899aa; font-size: 10px; }"
        "#userNameLabel { color: #2c3e50; font-size: 12px; font-weight: bold; }"

        // --- GroupBox 统一风格 ---
        "QGroupBox {"
        "  border: 1px solid #d4dce6; border-radius: 6px;"
        "  margin-top: 10px; font-weight: 600; font-size: 11px; color: #34495e;"
        "  background-color: #ffffff;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin; left: 10px; padding: 0 4px;"
        "  color: #34495e;"
        "}"

        // --- 字段标签 ---
        "#fieldLabel { color: #5d6d7e; font-size: 10px; }"

        // --- 输入控件 ---
        "QLineEdit, QComboBox, QDoubleSpinBox {"
        "  background-color: #f8fafc; border: 1px solid #c8d3df; border-radius: 4px;"
        "  padding: 2px 6px; min-height: 22px; color: #2c3e50; font-size: 11px;"
        "}"
        "QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border-color: #4a90c4; }"
        "QComboBox::drop-down { border: none; width: 20px; }"
        "QComboBox QAbstractItemView {"
        "  background-color: white; border: 1px solid #c8d3df;"
        "  selection-background-color: #4a90c4; selection-color: white; font-size: 11px;"
        "}"

        // --- 主操作按钮(蓝) ---
        "QPushButton#primaryButton {"
        "  background-color: #4a90c4; color: white; border: none; border-radius: 4px;"
        "  padding: 4px 10px; min-height: 24px; font-weight: 600; font-size: 11px;"
        "}"
        "QPushButton#primaryButton:hover { background-color: #3d7fb3; }"
        "QPushButton#primaryButton:pressed { background-color: #346fa0; }"

        // --- 次级操作按钮(灰蓝) ---
        "QPushButton#secondaryButton {"
        "  background-color: #8eafc5; color: white; border: none; border-radius: 4px;"
        "  padding: 2px 8px; min-height: 22px; font-size: 10px;"
        "}"
        "QPushButton#secondaryButton:hover { background-color: #7a9db5; }"

        // --- 强调按钮(绿色-开始检测) ---
        "QPushButton#accentButton {"
        "  background-color: #27ae60; color: white; border: none; border-radius: 3px;"
        "  padding: 2px 6px; min-height: 20px; font-weight: 600; font-size: 10px;"
        "}"
        "QPushButton#accentButton:hover { background-color: #219a52; }"
        "QPushButton#accentButton:pressed { background-color: #1e8449; }"

        // --- 危险按钮(红色-停止检测) ---
        "QPushButton#dangerButton {"
        "  background-color: #e74c3c; color: white; border: none; border-radius: 3px;"
        "  padding: 2px 6px; min-height: 20px; font-weight: 600; font-size: 10px;"
        "}"
        "QPushButton#dangerButton:hover { background-color: #c0392b; }"
        "QPushButton#dangerButton:pressed { background-color: #a93226; }"

        // --- 次要动作按钮(灰色-清除/导出) ---
        "QPushButton#secondaryAction {"
        "  background-color: #95a5a6; color: white; border: none; border-radius: 3px;"
        "  padding: 2px 6px; min-height: 20px; font-size: 10px;"
        "}"
        "QPushButton#secondaryAction:hover { background-color: #7f8c8d; }"
        "QPushButton#secondaryAction:pressed { background-color: #6c7a7b; }"

        // --- 退出登录按钮 ---
        "QPushButton#logoutButton {"
        "  background-color: transparent; color: #7f8c8d; border: 1px solid #bdc3c7;"
        "  border-radius: 4px; padding: 4px 10px; min-height: 24px; font-size: 11px;"
        "}"
        "QPushButton#logoutButton:hover { background-color: #f5f6f7; color: #e74c3c; border-color: #e74c3c; }"

        // --- Radio 按钮 ---
        "QRadioButton { spacing: 4px; color: #2c3e50; font-size: 11px; }"
        "QRadioButton::indicator { width: 14px; height: 14px; }"

        // --- 状态值标签 ---
        "#statusValue { color: #2c3e50; font-weight: 600; font-size: 11px; }"

        // --- 进度条 ---
        "QProgressBar#progressBar {"
        "  background-color: #e8ecf1; border: none; border-radius: 3px;"
        "  min-height: 14px; text-align: center; color: #5d6d7e; font-size: 10px;"
        "}"
        "QProgressBar#progressBar::chunk {"
        "  background-color: #4a90c4; border-radius: 3px;"
        "}"

        // --- 预览区 Tab ---
        "QTabWidget#previewTabs::pane { border: 1px solid #d4dce6; border-radius: 4px; background-color: #f8fafc; }"
        "QTabBar::tab {"
        "  background: #e4eaf0; color: #5d6d7e; padding: 6px 20px;"
        "  border-top-left-radius: 4px; border-top-right-radius: 4px;"
        "  margin-right: 2px; font-size: 11px; min-width: 60px;"
        "}"
        "QTabBar::tab:selected { background: #ffffff; color: #2c3e50; font-weight: 600; }"
        "QTabBar::tab:hover { background: #dce4ed; }"

        // --- 图像预览标签 ---
        "#imageLabel { background-color: #f0f3f7; border: 1px dashed #c8d3df; border-radius: 4px; color: #8899aa; font-size: 12px; }"

        // --- 摘要文本 ---
        "#summaryText { background-color: #f8fafc; border: 1px solid #d4dce6; border-radius: 4px; color: #5d6d7e; font-size: 11px; }"

        // --- 日志区域 ---
        "#logEdit { background-color: #1e2a36; color: #a8d8a8; border: none; border-radius: 4px; font-family: 'Consolas', 'Menlo', monospace; font-size: 11px; padding: 4px; }"

        // --- 表格 ---
        "QTableWidget {"
        "  background-color: white; border: 1px solid #d4dce6; border-radius: 4px;"
        "  gridline-color: #e8ecf1; color: #2c3e50; font-size: 11px;"
        "}"
        "QTableWidget::item { padding: 2px 6px; }"
        "QTableWidget::item:selected { background-color: #4a90c4; color: white; }"
        "QHeaderView::section {"
        "  background-color: #f0f3f7; color: #34495e; font-weight: 600;"
        "  border: none; border-bottom: 2px solid #d4dce6; padding: 4px 6px; font-size: 11px;"
        "}"
        "QTableWidget QTableCornerButton::section { background-color: #f0f3f7; border: none; }"

        // --- TextEdit ---
        "QTextEdit { background-color: #f8fafc; border: 1px solid #c8d3df; border-radius: 4px; color: #2c3e50; }"

        // --- PlainTextEdit ---
        "QPlainTextEdit { border: 1px solid #d4dce6; border-radius: 4px; }"

        // --- 按钮通用禁用态 ---
        "QPushButton:disabled { background-color: #bdc3c7; color: #ecf0f1; }"
    ));
}

void DetectionWidget::appendLog(const QString& text) {
    if (m_logEdit) {
        m_logEdit->appendPlainText(withTimestamp(text));
    }
}

void DetectionWidget::updateStatusDisplay(const QString& detectionStatus) {
    m_detectionStatusValue->setText(detectionStatus);
    emit statusMessageChanged(detectionStatus);
}

void DetectionWidget::updateImagePreview(QLabel* label, const QImage& image) {
    if (image.isNull()) {
        label->clear();
        label->setText(QStringLiteral("暂无图像"));
        return;
    }

    // 使用 QLabel 实际显示尺寸 × devicePixelRatio 保证高清渲染
    const qreal dpr = devicePixelRatioF();
    const QSize targetSize = label->size() * dpr;
    const QPixmap pixmap = QPixmap::fromImage(image).scaled(targetSize,
                                                            Qt::KeepAspectRatio,
                                                            Qt::SmoothTransformation);
    // 设置 pixmap 的 devicePixelRatio 让 QLabel 按逻辑尺寸显示
    QPixmap scaled = pixmap;
    scaled.setDevicePixelRatio(dpr);
    label->setPixmap(scaled);
}

void DetectionWidget::populateTable(const QList<IndustryVision::DetectionResult>& results) {
    m_resultTable->setRowCount(results.size());

    for (int row = 0; row < results.size(); ++row) {
        const IndustryVision::DetectionResult& result = results.at(row);
        m_resultTable->setItem(row, 0, new QTableWidgetItem(result.className));
        m_resultTable->setItem(row, 1, new QTableWidgetItem(QString::number(result.confidence, 'f', 2)));
        m_resultTable->setItem(row, 2, new QTableWidgetItem(QStringLiteral("(%1, %2)")
                                                                .arg(result.boundingBox.x())
                                                                .arg(result.boundingBox.y())));
        m_resultTable->setItem(row, 3, new QTableWidgetItem(QStringLiteral("%1 x %2")
                                                                .arg(result.boundingBox.width())
                                                                .arg(result.boundingBox.height())));
    }
}

QImage DetectionWidget::createPlaceholderFrame(const QString& title, const QString& subtitle, int frameIndex) const {
    // 视频/摄像头模式当前提供统一占位帧，方便后续接入真实流媒体后端。
    QImage image(960, 720, QImage::Format_ARGB32_Premultiplied);
    image.fill(QColor(228, 235, 242));

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const QRect panelRect(56, 56, image.width() - 112, image.height() - 112);
    painter.fillRect(panelRect, QColor(248, 250, 252));
    painter.setPen(QPen(QColor(78, 147, 200), 3));
    painter.drawRect(panelRect);

    painter.setPen(QColor(42, 61, 77));
    QFont titleFont = painter.font();
    titleFont.setPointSize(28);
    titleFont.setBold(true);
    painter.setFont(titleFont);
    painter.drawText(panelRect.adjusted(30, 60, -30, -30), title);

    QFont subtitleFont = painter.font();
    subtitleFont.setPointSize(18);
    subtitleFont.setBold(false);
    painter.setFont(subtitleFont);
    painter.drawText(panelRect.adjusted(30, 120, -30, -30), subtitle);

    painter.drawText(panelRect.adjusted(30, 170, -30, -30), QStringLiteral("Frame #%1").arg(frameIndex));
    painter.drawText(panelRect.adjusted(30, 220, -30, -30), QStringLiteral("Version: %1").arg(m_versionCombo->currentText()));
    painter.drawText(panelRect.adjusted(30, 270, -30, -30),
                     QStringLiteral("Confidence %1 | IOU %2")
                         .arg(m_confidenceSpin->value(), 0, 'f', 2)
                         .arg(m_iouSpin->value(), 0, 'f', 2));

    painter.end();
    return image;
}

void DetectionWidget::performDetection(const QImage& image, const QString& sourceName) {
    if (image.isNull()) {
        appendLog(QStringLiteral("检测失败：无法读取输入源。"));
        updateStatusDisplay(QStringLiteral("输入无效"));
        return;
    }

    m_lastOriginalImage = image;
    updateImagePreview(m_originalImageLabel, m_lastOriginalImage);
    m_progressBar->setValue(55);

    const IndustryVision::DetectionReport report = m_engine.detect(image, sourceName);
    m_lastAnnotatedImage = report.annotatedImage;
    m_lastResults = report.results;

    updateImagePreview(m_resultImageLabel, m_lastAnnotatedImage);
    populateTable(report.results);
    m_summaryText->setText(report.summaryText);
    m_currentSourceValue->setText(sourceName);
    m_progressBar->setValue(100);
    updateStatusDisplay(QStringLiteral("检测完成"));

    appendLog(QStringLiteral("检测完成：%1").arg(sourceName));
    appendLog(report.summaryText);
}

IndustryVision::InputSourceMode DetectionWidget::currentMode() const {
    if (m_videoModeButton->isChecked()) {
        return IndustryVision::InputSourceMode::Video;
    }

    if (m_cameraModeButton->isChecked()) {
        return IndustryVision::InputSourceMode::Camera;
    }

    return IndustryVision::InputSourceMode::Image;
}

} // namespace IndustryVision::Gui
