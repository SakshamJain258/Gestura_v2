"""
app.py — Entry Point

This file is intentionally minimal. Its only job is:
1. Create the Qt application
2. Create the main window
3. Start the event loop

All the real logic lives in ui/main_window.py, threads/, and core/.

HOW TO RUN:
    python app.py

TO CHANGE CAMERA:
    Edit the camera_index in ui/main_window.py → MainWindow.__init__
    Common values:
      0 = built-in webcam
      1 = external USB webcam
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Gestura")

    window = MainWindow()
    window.show()

    # sys.exit ensures Python exits with the right code
    # when the Qt event loop ends
    sys.exit(app.exec())


if __name__ == "__main__":
    main()