#include "IndustryVisionGUI/ApplicationWindow.h"

#include <QStackedWidget>
#include <QStatusBar>
#include <QScreen>
#include <QGuiApplication>

namespace IndustryVision::Gui {

ApplicationWindow::ApplicationWindow(QWidget* parent)
    : QMainWindow(parent)
    , m_stack(new QStackedWidget(this))
    , m_loginWidget(new LoginWidget(this))
    , m_detectionWidget(new DetectionWidget(this)) {
    setWindowTitle(QStringLiteral("通用检测系统"));
    setCentralWidget(m_stack);

    setStyleSheet(QStringLiteral(
        "QMainWindow { background-color: #edf1f5; }"
        "QStatusBar { background-color: #f8fafc; color: #5d6d7e; border-top: 1px solid #d4dce6; font-size: 12px; padding: 2px 8px; }"
    ));
    statusBar()->showMessage(QStringLiteral("系统已就绪"));

    m_stack->addWidget(m_loginWidget);
    m_stack->addWidget(m_detectionWidget);
    m_stack->setCurrentWidget(m_loginWidget);

    connect(m_loginWidget, &LoginWidget::loginSucceeded, this, &ApplicationWindow::showDetectionView);
    connect(m_detectionWidget, &DetectionWidget::logoutRequested, this, &ApplicationWindow::showLoginView);
    connect(m_detectionWidget, &DetectionWidget::statusMessageChanged, this, [this](const QString& message) {
        statusBar()->showMessage(message);
    });

    showLoginView();
}

void ApplicationWindow::showDetectionView(const QString& username) {
    m_detectionWidget->setCurrentUser(username);
    m_stack->setCurrentWidget(m_detectionWidget);

    const int width = 1340;
    const int height = 860;
    resize(width, height);

    setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    setMinimumSize(1024, 680);

    QScreen* screen = QGuiApplication::primaryScreen();
    const QRect availableGeometry = screen->availableGeometry();
    const int x = availableGeometry.left() + (availableGeometry.width() - width) / 2;
    const int y = availableGeometry.top() + (availableGeometry.height() - height) / 2;
    move(x, y);

    statusBar()->showMessage(QStringLiteral("欢迎回来，%1").arg(username));
}

void ApplicationWindow::showLoginView() {
    m_stack->setCurrentWidget(m_loginWidget);

    const int loginWidth = 420;
    const int loginHeight = 440;
    resize(loginWidth, loginHeight);

    QScreen* screen = QGuiApplication::primaryScreen();
    const QRect availableGeometry = screen->availableGeometry();
    const int x = availableGeometry.left() + (availableGeometry.width() - loginWidth) / 2;
    const int y = availableGeometry.top() + (availableGeometry.height() - loginHeight) / 2;
    move(x, y);

    setMaximumSize(loginWidth, loginHeight);
    setMinimumSize(loginWidth, loginHeight);

    statusBar()->showMessage(QStringLiteral("请登录以继续"));
}

} // namespace IndustryVision::Gui
