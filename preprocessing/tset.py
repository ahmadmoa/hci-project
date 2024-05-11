import mne
import numpy as np

from read import load_data
from preprocess import preprocess_data

def load_and_preprocess(file):
    raw = load_data(file)
    return preprocess_data(raw)

def extract_data_and_labels(raw):
    events, _ = mne.events_from_annotations(raw)
    print("events", events)
    epochs = mne.Epochs(raw, events, event_id={'left_hand': 769, 'right_hand': 770}, tmin=-0.2, tmax=0.8, baseline=None)
    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1] - 769
    return epochs_data, labels

def load_data_from_files(file_list):
    all_data = []
    all_labels = []
    for file in file_list:
        raw = load_and_preprocess(file)
        data, labels = extract_data_and_labels(raw)
        all_data.append(data)
        all_labels.append(labels)
    return np.concatenate(all_data), np.concatenate(all_labels)

if __name__ == '__main__':
    # List of training and evaluation files
    training_files = ['../data/A01T.gdf', '../data/A02T.gdf']
    evaluation_files = ['../data/A01E.gdf', '../data/A02E.gdf']
    training_data, training_labels = load_data_from_files(training_files)
    print(training_data)
    print(training_labels)