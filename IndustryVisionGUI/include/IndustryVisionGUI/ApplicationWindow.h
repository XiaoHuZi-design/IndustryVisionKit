#pragma once

#include "IndustryVisionGUI/DetectionWidget.h"
#include "IndustryVisionGUI/LoginWidget.h"

#include <QMainWindow>

QT_BEGIN_NAMESPACE
class QStackedWidget;
QT_END_NAMESPACE

namespace IndustryVision::Gui {

class ApplicationWindow final : public QMainWindow {
    Q_OBJECT

public:
    explicit ApplicationWindow(QWidget* parent = nullptr);

private slots:
    void showDetectionView(const QString& username);
    void showLoginView();

private:
    QStackedWidget* m_stack = nullptr;
    LoginWidget* m_loginWidget = nullptr;
    DetectionWidget* m_detectionWidget = nullptr;
};

} // namespace IndustryVision::Gui
