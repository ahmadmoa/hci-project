from preprocessing.preprocessing import preprocess_data, load_and_preprocess
from preprocessing.read import load_data
from models.modle import extract_features
import mne
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score



if __name__ == '__main__':
    evaluation_files = ["data/A01E.gdf"]
    evaluation_epochs_data = []
    evaluation_labels = []

    for file_path in evaluation_files:
        raw = load_and_preprocess(file_path)
        events, event_id = mne.events_from_annotations(raw)

        print(f"Processing file: {file_path}")
        print(f"Event IDs: {event_id}")
        print(f"Event Descriptions: {events}")

        # Use event descriptions to identify the correct IDs
        left_hand_event_id = event_id.get('768', 0)
        right_hand_event_id = event_id.get('783', 0)

        if left_hand_event_id == 0 or right_hand_event_id == 0:
            print(f"Required event IDs (768, 783) not found in the data for file {file_path}.")
            continue

        epochs = mne.Epochs(raw, events, event_id={'left_hand': left_hand_event_id, 'right_hand': right_hand_event_id}, tmin=-0.2, tmax=0.8, baseline=None)
        epochs_data = epochs.get_data()
        labels = epochs.events[:, -1] - left_hand_event_id  # Adjust to match event IDs

        evaluation_epochs_data.append(epochs_data)
        evaluation_labels.append(labels)

    if evaluation_epochs_data and evaluation_labels:
        evaluation_epochs_data = np.concatenate(evaluation_epochs_data, axis=0)
        evaluation_labels = np.concatenate(evaluation_labels, axis=0)

        evaluation_features = extract_features(evaluation_epochs_data, evaluation_labels)

        # Load the trained models
        svm = load('svm_model.joblib')
        lda = load('lda_model.joblib')

        svm_predictions = svm.predict(evaluation_features)
        lda_predictions = lda.predict(evaluation_features)

        # Evaluate the predictions (if true labels are available)
        svm_evaluation_score = accuracy_score(evaluation_labels, svm_predictions)
        lda_evaluation_score = accuracy_score(evaluation_labels, lda_predictions)

        print(f'SVM Evaluation Accuracy: {svm_evaluation_score}')
        print(f'LDA Evaluation Accuracy: {lda_evaluation_score}')
    else:
        print("No valid evaluation data found.")
