#include "IndustryVisionGUI/ApplicationWindow.h"

#include <QApplication>
#include <cstdlib>

int main(int argc, char* argv[]) {
    // LibTorch 的 OpenMP 运行时可能和系统已有库冲突，设置此环境变量跳过检查
    setenv("KMP_DUPLICATE_LIB_OK", "TRUE", 1);

    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("IndustryVisionKit"));
    app.setOrganizationName(QStringLiteral("IndustryVisionKit"));

    IndustryVision::Gui::ApplicationWindow window;
    window.show();

    return app.exec();
}
