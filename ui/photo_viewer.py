import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, \
    QComboBox, QScrollArea, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from joblib import load
import mne
import numpy as np
from preprocessing.preprocessing import preprocess_data
from preprocessing.read import load_data
from models.model import extract_features


class PhotoViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.svm_model = load('svm_model.joblib')
        self.lda_model = load('lda_model.joblib')

    def initUI(self):
        self.photos = ['./data/images/1.png', './data/images/2.png', './data/images/3.png', './data/images/4.png', './data/images/5.png']
        self.current_index = len(self.photos) // 2  # Center the initial selection

        # Create widgets
        self.label = QLabel(self)
        self.pixmap = QPixmap(self.photos[self.current_index])
        self.label.setPixmap(self.pixmap)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(800, 600)  # Fixed size for the image label
        self.label.setStyleSheet("border: 2px solid #444; padding: 10px; background-color: #f0f0f0;")

        self.left_button = QPushButton('<', self)
        self.right_button = QPushButton('>', self)
        self.browse_button = QPushButton('Browse Evaluation File', self)
        self.model_selector = QComboBox(self)
        self.model_selector.addItems(['SVM', 'LDA'])

        # Set fonts
        button_font = QFont('Arial', 14, QFont.Bold)
        self.left_button.setFont(button_font)
        self.right_button.setFont(button_font)
        self.browse_button.setFont(QFont('Arial', 12))
        self.model_selector.setFont(QFont('Arial', 12))

        # Set styles
        self.left_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 10px; padding: 10px;")
        self.right_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 10px; padding: 10px;")
        self.browse_button.setStyleSheet("background-color: #008CBA; color: white; border-radius: 10px; padding: 10px;")
        self.model_selector.setStyleSheet("padding: 5px; border-radius: 5px;")

        # Connect signals
        self.left_button.clicked.connect(self.show_prev_photo)
        self.right_button.clicked.connect(self.show_next_photo)
        self.browse_button.clicked.connect(self.browse_file)

        # Layouts for navigation buttons
        hbox = QHBoxLayout()
        hbox.addWidget(self.left_button)
        hbox.addWidget(self.right_button)

        # Layout for thumbnail slider
        self.slider = QScrollArea()
        self.slider.setFixedHeight(120)
        self.slider.setWidgetResizable(True)
        self.thumb_widget = QWidget()
        self.thumb_layout = QHBoxLayout(self.thumb_widget)
        self.slider.setWidget(self.thumb_widget)
        self.update_thumbnails()

        # Main layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.browse_button)
        vbox.addWidget(self.model_selector)
        vbox.addLayout(hbox)
        vbox.addWidget(self.slider)

        self.setLayout(vbox)
        self.setWindowTitle('Photo Viewer')
        self.setFixedSize(1000, 800)  # Fixed window size
        self.setStyleSheet("background-color: #f5f5f5;")
        self.show()

        # Center the initial selection
        QTimer.singleShot(100, self.center_initial_selection)

    def center_initial_selection(self):
        # Calculate the position to center the selected thumbnail
        if self.current_index < len(self.photos):
            selected_thumbnail = self.thumb_layout.itemAt(self.current_index).widget()
            if selected_thumbnail:
                scroll_area_width = self.slider.viewport().width()
                thumbnail_width = selected_thumbnail.width()
                offset = selected_thumbnail.pos().x() - (scroll_area_width // 2) + (thumbnail_width // 2)
                self.slider.horizontalScrollBar().setValue(offset)

    def update_thumbnails(self):
        # Clear existing thumbnails
        while self.thumb_layout.count():
            item = self.thumb_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for i, photo in enumerate(self.photos):
            thumbnail = QLabel(self)
            pixmap = QPixmap(photo).scaled(100, 75, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            thumbnail.setPixmap(pixmap)
            thumbnail.setStyleSheet("border: 2px solid #ddd;")
            if i == self.current_index:
                thumbnail.setStyleSheet("border: 2px solid #4CAF50;")
            thumbnail.mousePressEvent = lambda event, index=i: self.set_current_index(index)
            self.thumb_layout.addWidget(thumbnail)
        self.thumb_widget.setLayout(self.thumb_layout)

    def set_current_index(self, index):
        self.current_index = index
        self.update_photo()

    def browse_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Open Evaluation File', '', 'GDF Files (*.gdf)')
        if file_path:
            self.evaluate_file(file_path)

    def evaluate_file(self, file_path):
        model_type = self.model_selector.currentText()
        print(f"Using model: {model_type}")

        raw = load_data(file_path)
        raw = preprocess_data(raw)
        events, event_id = mne.events_from_annotations(raw)

        left_hand_event_id = event_id.get('768', 0)
        right_hand_event_id = event_id.get('783', 0)

        if left_hand_event_id == 0 or right_hand_event_id == 0:
            print(f"Required event IDs (768, 783) not found in the data for file {file_path}.")
            return

        epochs = mne.Epochs(raw, events, event_id={'left_hand': left_hand_event_id, 'right_hand': right_hand_event_id},
                            tmin=-0.2, tmax=0.8, baseline=None)
        epochs_data = epochs.get_data()
        labels = epochs.events[:, -1] - left_hand_event_id

        features = extract_features(epochs_data, labels)

        if model_type == 'SVM':
            predictions = self.svm_model.predict(features)
        elif model_type == 'LDA':
            predictions = self.lda_model.predict(features)
        else:
            print(f"Unknown model type: {model_type}")
            return

        print(f"File: {file_path}, {model_type} Predictions: {predictions}")

        self.process_predictions(predictions)

    def process_predictions(self, predictions):
        print(f"Processing {len(predictions)} predictions.")
        for prediction in predictions:
            if prediction == 0:  # left hand
                self.show_prev_photo()
            elif prediction == 1:  # right hand
                self.show_next_photo()

    def show_prev_photo(self):
        if self.current_index > 0:
            self.set_current_index(self.current_index - 1)

    def show_next_photo(self):
        if self.current_index < len(self.photos) - 1:
            self.set_current_index(self.current_index + 1)

    def update_photo(self):
        self.pixmap = QPixmap(self.photos[self.current_index])
        self.label.setPixmap(self.pixmap)
        self.update_thumbnails()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = PhotoViewer()
    sys.exit(app.exec_())
