import sys
from PyQt5.QtWidgets import QApplication
from ui.photo_viewer import PhotoViewer

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = PhotoViewer()
    sys.exit(app.exec_())
