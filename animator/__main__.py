"""Start the app: python -m animator"""
import os
import sys

os.environ.setdefault("GLOG_minloglevel", "2")       # quieten MediaPipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main():
    from PySide6.QtWidgets import QApplication

    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
