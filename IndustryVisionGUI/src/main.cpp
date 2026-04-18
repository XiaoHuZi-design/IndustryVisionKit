#include "IndustryVisionGUI/ApplicationWindow.h"

#include <QApplication>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("IndustryVisionKit"));
    app.setOrganizationName(QStringLiteral("IndustryVisionKit"));

    IndustryVision::Gui::ApplicationWindow window;
    window.show();

    return app.exec();
}
