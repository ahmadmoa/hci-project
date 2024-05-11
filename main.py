from preprocessing.preprocess import preprocess_data
from preprocessing.read import load_data
from models.modle import extract_features, train_svm, train_lda
from ui.photo_viewer import PhotoViewer
import mne

if __name__ == '__main__':
    # Load and preprocess EEG data
    raw = load_data("data/A01T.gdf")
    raw = preprocess_data(raw)

    # Create epochs and labels
    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id={'left_hand': 769, 'right_hand': 770}, tmin=-0.2, tmax=0.8, baseline=None)
    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1] - 769

    # Feature extraction
    features = extract_features(epochs_data, labels)

    # Train classifiers
    svm, svm_score = train_svm(features, labels)
    lda, lda_score = train_lda(features, labels)

    print(f'SVM Accuracy: {svm_score}')
    print(f'LDA Accuracy: {lda_score}')

    # Start the UI
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    viewer = PhotoViewer()
    sys.exit(app.exec_())