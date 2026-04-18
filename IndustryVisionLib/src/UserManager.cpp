#include "IndustryVisionLib/UserManager.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStandardPaths>

namespace IndustryVision {

namespace {

QString hashPassword(const QString& password) {
    return QString::fromUtf8(QCryptographicHash::hash(password.toUtf8(), QCryptographicHash::Sha256).toHex());
}

} // namespace

UserManager::UserManager() = default;

bool UserManager::registerUser(const QString& username, const QString& password, QString* message) {
    if (!validateCredentials(username, password, message)) {
        return false;
    }

    QFile file(userStorePath());
    QJsonObject usersObject;

    if (file.exists() && file.open(QIODevice::ReadOnly)) {
        const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
        usersObject = document.object();
        file.close();
    }

    if (usersObject.contains(username)) {
        if (message != nullptr) {
            *message = QStringLiteral("用户名已存在，请直接登录。");
        }
        return false;
    }

    usersObject.insert(username, hashPassword(password));

    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (message != nullptr) {
            *message = QStringLiteral("用户信息保存失败。");
        }
        return false;
    }

    file.write(QJsonDocument(usersObject).toJson(QJsonDocument::Indented));
    file.close();

    if (message != nullptr) {
        *message = QStringLiteral("注册成功，请使用新账号登录。");
    }
    return true;
}

bool UserManager::login(const QString& username, const QString& password, QString* message) const {
    if (!validateCredentials(username, password, message)) {
        return false;
    }

    QFile file(userStorePath());
    if (!file.exists()) {
        if (message != nullptr) {
            *message = QStringLiteral("未找到用户信息，请先注册。");
        }
        return false;
    }

    if (!file.open(QIODevice::ReadOnly)) {
        if (message != nullptr) {
            *message = QStringLiteral("用户信息读取失败。");
        }
        return false;
    }

    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    const QJsonObject usersObject = document.object();
    file.close();

    if (!usersObject.contains(username)) {
        if (message != nullptr) {
            *message = QStringLiteral("账号不存在，请先注册。");
        }
        return false;
    }

    if (usersObject.value(username).toString() != hashPassword(password)) {
        if (message != nullptr) {
            *message = QStringLiteral("密码错误，请重试。");
        }
        return false;
    }

    if (message != nullptr) {
        *message = QStringLiteral("登录成功。");
    }
    return true;
}

QString UserManager::userStorePath() const {
    // 用户信息默认存储到应用数据目录，避免写入源码目录。
    const QString appDataPath =
        QStandardPaths::writableLocation(QStandardPaths::AppDataLocation).isEmpty()
            ? QDir::currentPath() + QStringLiteral("/users")
            : QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);

    QDir dir(appDataPath);
    if (!dir.exists()) {
        dir.mkpath(QStringLiteral("."));
    }

    return dir.filePath(QStringLiteral("users.json"));
}

bool UserManager::validateCredentials(const QString& username, const QString& password, QString* message) const {
    if (username.trimmed().size() < 3) {
        if (message != nullptr) {
            *message = QStringLiteral("用户名至少为 3 位字符。");
        }
        return false;
    }

    if (password.size() < 6) {
        if (message != nullptr) {
            *message = QStringLiteral("密码至少为 6 位字符。");
        }
        return false;
    }

    return true;
}

} // namespace IndustryVision
