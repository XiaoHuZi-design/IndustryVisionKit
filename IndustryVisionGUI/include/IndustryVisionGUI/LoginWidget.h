#pragma once

#include "IndustryVisionLib/UserManager.h"

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QPushButton;
QT_END_NAMESPACE

namespace IndustryVision::Gui {

class LoginWidget final : public QWidget {
    Q_OBJECT

public:
    explicit LoginWidget(QWidget* parent = nullptr);

signals:
    void loginSucceeded(const QString& username);

private slots:
    void handleLogin();
    void handleRegister();

private:
    void buildUi();
    void updateMessage(const QString& text, bool success);

    IndustryVision::UserManager m_userManager;
    QLineEdit* m_usernameEdit = nullptr;
    QLineEdit* m_passwordEdit = nullptr;
    QLabel* m_messageLabel = nullptr;
    QPushButton* m_loginButton = nullptr;
    QPushButton* m_registerButton = nullptr;
};

} // namespace IndustryVision::Gui
