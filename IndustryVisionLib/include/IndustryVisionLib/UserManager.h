#pragma once

#include <QString>

namespace IndustryVision {

class UserManager {
public:
    UserManager();

    bool registerUser(const QString& username, const QString& password, QString* message);
    bool login(const QString& username, const QString& password, QString* message) const;

private:
    QString userStorePath() const;
    bool validateCredentials(const QString& username, const QString& password, QString* message) const;
};

} // namespace IndustryVision
