#include "IndustryVisionGUI/LoginWidget.h"

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

namespace IndustryVision::Gui {

LoginWidget::LoginWidget(QWidget* parent)
    : QWidget(parent) {
    buildUi();
}

void LoginWidget::handleLogin() {
    QString message;
    const bool ok = m_userManager.login(m_usernameEdit->text().trimmed(), m_passwordEdit->text(), &message);
    updateMessage(message, ok);

    if (ok) {
        QMessageBox::information(this,
                                 QStringLiteral("欢迎"),
                                 QStringLiteral("欢迎回来，%1！").arg(m_usernameEdit->text().trimmed()));
        emit loginSucceeded(m_usernameEdit->text().trimmed());
    }
}

void LoginWidget::handleRegister() {
    QString message;
    const bool ok = m_userManager.registerUser(m_usernameEdit->text().trimmed(), m_passwordEdit->text(), &message);
    updateMessage(message, ok);
}

void LoginWidget::buildUi() {
    setObjectName(QStringLiteral("loginRoot"));
    setStyleSheet(QStringLiteral(
        // --- 背景 ---
        "#loginRoot { background-color: #edf1f5; }"
        // --- 卡片容器 ---
        "QFrame#loginCard {"
        "  background-color: #ffffff; border: 1px solid #d4dce6;"
        "  border-radius: 12px;"
        "}"
        // --- 标题 ---
        "QLabel#titleLabel {"
        "  font-size: 28px; font-weight: 700; color: #2c3e50;"
        "  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;"
        "}"
        // --- 提示信息 ---
        "QLabel#messageLabel {"
        "  font-size: 12px; padding: 8px 12px; border-radius: 6px;"
        "  background-color: #edf5ff; color: #4a90c4;"
        "}"
        // --- 输入框 ---
        "QLineEdit {"
        "  min-height: 36px; border: 1px solid #c8d3df; border-radius: 6px;"
        "  padding: 0 12px; background-color: #f8fafc; color: #2c3e50;"
        "  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;"
        "}"
        "QLineEdit:focus { border-color: #4a90c4; }"
        // --- 按钮 ---
        "QPushButton#loginBtn {"
        "  min-height: 38px; border-radius: 6px; color: white;"
        "  background-color: #4a90c4; font-size: 14px; font-weight: 600;"
        "  border: none;"
        "}"
        "QPushButton#loginBtn:hover { background-color: #3d7fb3; }"
        "QPushButton#registerBtn {"
        "  min-height: 38px; border-radius: 6px; color: #4a90c4;"
        "  background-color: transparent; font-size: 14px;"
        "  border: 1px solid #4a90c4;"
        "}"
        "QPushButton#registerBtn:hover { background-color: #f0f7fc; }"
    ));

    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(24, 24, 24, 24);

    auto* card = new QFrame(this);
    card->setObjectName(QStringLiteral("loginCard"));
    rootLayout->addWidget(card);

    auto* cardLayout = new QVBoxLayout(card);
    cardLayout->setContentsMargins(28, 28, 28, 20);
    cardLayout->setSpacing(0);

    auto* titleLabel = new QLabel(QStringLiteral("通用检测系统"), card);
    titleLabel->setObjectName(QStringLiteral("titleLabel"));
    titleLabel->setAlignment(Qt::AlignCenter);

    auto* sectionLabel = new QLabel(QStringLiteral("用户登录"), card);
    sectionLabel->setStyleSheet(QStringLiteral(
        "font-size: 16px; font-weight: 600; color: #5d6d7e;"
        "font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;"
    ));

    m_messageLabel = new QLabel(QStringLiteral("请输入账号与密码，首次使用可直接注册。"), card);
    m_messageLabel->setObjectName(QStringLiteral("messageLabel"));
    m_messageLabel->setWordWrap(true);
    m_messageLabel->setFixedHeight(36);

    auto* usernameLabel = new QLabel(QStringLiteral("用户名"), card);
    usernameLabel->setStyleSheet(QStringLiteral("color: #5d6d7e; font-size: 12px; font-weight: 600;"));

    m_usernameEdit = new QLineEdit(card);
    m_usernameEdit->setPlaceholderText(QStringLiteral("请输入用户名"));

    auto* passwordLabel = new QLabel(QStringLiteral("密码"), card);
    passwordLabel->setStyleSheet(QStringLiteral("color: #5d6d7e; font-size: 12px; font-weight: 600;"));

    m_passwordEdit = new QLineEdit(card);
    m_passwordEdit->setPlaceholderText(QStringLiteral("请输入密码"));
    m_passwordEdit->setEchoMode(QLineEdit::Password);

    // 用户名字段组：标签紧贴输入框
    auto* usernameFieldLayout = new QVBoxLayout();
    usernameFieldLayout->setSpacing(4);
    usernameFieldLayout->addWidget(usernameLabel);
    usernameFieldLayout->addWidget(m_usernameEdit);

    // 密码字段组：标签紧贴输入框
    auto* passwordFieldLayout = new QVBoxLayout();
    passwordFieldLayout->setSpacing(4);
    passwordFieldLayout->addWidget(passwordLabel);
    passwordFieldLayout->addWidget(m_passwordEdit);

    m_loginButton = new QPushButton(QStringLiteral("登  录"), card);
    m_loginButton->setObjectName(QStringLiteral("loginBtn"));
    m_registerButton = new QPushButton(QStringLiteral("注  册"), card);
    m_registerButton->setObjectName(QStringLiteral("registerBtn"));

    auto* buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(10);
    buttonLayout->addWidget(m_loginButton, 1);
    buttonLayout->addWidget(m_registerButton, 1);

    auto* footerLabel = new QLabel(QStringLiteral("v1.0.0  \302\251 2026 通用检测系统"), card);
    footerLabel->setAlignment(Qt::AlignCenter);
    footerLabel->setStyleSheet(QStringLiteral("color: #95a5a6; font-size: 11px;"));

    cardLayout->addWidget(titleLabel);
    cardLayout->addSpacing(4);
    cardLayout->addWidget(sectionLabel);
    cardLayout->addSpacing(8);
    cardLayout->addWidget(m_messageLabel);
    cardLayout->addSpacing(16);
    cardLayout->addLayout(usernameFieldLayout);
    cardLayout->addSpacing(14);
    cardLayout->addLayout(passwordFieldLayout);
    cardLayout->addSpacing(20);
    cardLayout->addLayout(buttonLayout);
    cardLayout->addStretch();
    cardLayout->addSpacing(8);
    cardLayout->addWidget(footerLabel);

    connect(m_loginButton, &QPushButton::clicked, this, &LoginWidget::handleLogin);
    connect(m_registerButton, &QPushButton::clicked, this, &LoginWidget::handleRegister);
}

void LoginWidget::updateMessage(const QString& text, const bool success) {
    m_messageLabel->setText(text);
    m_messageLabel->setStyleSheet(success
                                      ? QStringLiteral("font-size: 12px; padding: 8px 12px; border-radius: 6px; background-color: #eafaf1; color: #27ae60;")
                                      : QStringLiteral("font-size: 12px; padding: 8px 12px; border-radius: 6px; background-color: #fdedec; color: #e74c3c;"));
}

} // namespace IndustryVision::Gui
