from preprocessing.preprocess import preprocess_data
from preprocessing.read import load_data
from models.modle import extract_features, train_svm, train_lda
from ui.photo_viewer import PhotoViewer
import mne
import os
import numpy as np


def load_and_preprocess(file_path):
    raw = load_data(file_path)
    raw = preprocess_data(raw)
    return raw


if __name__ == '__main__':


    train_files = ["data/A01T.gdf", "data/A02T.gdf", "data/A03T.gdf", "data/A04T.gdf", "data/A05T.gdf", "data/A06T.gdf",
                   "data/A07T.gdf", "data/A08T.gdf", "data/A09T.gdf"]

    all_epochs_data = []
    all_labels = []

    for file_path in train_files:
        raw = load_and_preprocess(file_path)


        events, event_id = mne.events_from_annotations(raw)


        if 7 not in event_id.values() or 8 not in event_id.values():
            print(f"Required event IDs (7, 8) not found in the data for file {file_path}.")
            continue


        epochs = mne.Epochs(raw, events, event_id={'left_hand': 7, 'right_hand': 8}, tmin=-0.2, tmax=0.8, baseline=None)
        epochs_data = epochs.get_data()
        labels = epochs.events[:, -1] - 7

        all_epochs_data.append(epochs_data)
        all_labels.append(labels)


    all_epochs_data = np.concatenate(all_epochs_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)


    features = extract_features(all_epochs_data, all_labels)


    svm, svm_score = train_svm(features, all_labels)
    lda, lda_score = train_lda(features, all_labels)

    print(f'SVM Accuracy: {svm_score}')
    print(f'LDA Accuracy: {lda_score}')


    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    viewer = PhotoViewer()
    sys.exit(app.exec_())
