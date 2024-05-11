import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap


class PhotoViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.photos = ['path/to/photo1.jpg', 'path/to/photo2.jpg', 'path/to/photo3.jpg']
        self.current_index = 0

        self.label = QLabel(self)
        self.pixmap = QPixmap(self.photos[self.current_index])
        self.label.setPixmap(self.pixmap)

        self.left_button = QPushButton('<', self)
        self.right_button = QPushButton('>', self)

        self.left_button.clicked.connect(self.show_prev_photo)
        self.right_button.clicked.connect(self.show_next_photo)

        hbox = QHBoxLayout()
        hbox.addWidget(self.left_button)
        hbox.addWidget(self.right_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addLayout(hbox)

        self.setLayout(vbox)
        self.setWindowTitle('Photo Viewer')
        self.show()

    def show_prev_photo(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_photo()

    def show_next_photo(self):
        if self.current_index < len(self.photos) - 1:
            self.current_index += 1
            self.update_photo()

    def update_photo(self):
        self.pixmap = QPixmap(self.photos[self.current_index])
        self.label.setPixmap(self.pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = PhotoViewer()
    sys.exit(app.exec_())