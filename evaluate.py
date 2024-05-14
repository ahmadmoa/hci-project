from preprocessing.preprocessing import load_and_preprocess
from models.model import apply_ica, extract_features_csp_pca
import mne
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score

def evaluate_model(file_path, apply_ica_flag=False):
    raw = load_and_preprocess(file_path)

    if apply_ica_flag:
        # Specify EOG channels if known, otherwise leave it as None
        eog_channels = ['EOG-left', 'EOG-central', 'EOG-right']
        raw = apply_ica(raw, eog_channels=eog_channels)

    events, event_id = mne.events_from_annotations(raw)

    print(f"Processing file: {file_path}")
    print(f"Event IDs: {event_id}")
    print(f"Event Descriptions: {events}")

    # Use event descriptions to identify the correct IDs
    left_hand_event_id = event_id.get('769', 0)
    right_hand_event_id = event_id.get('770', 0)

    if left_hand_event_id == 0 or right_hand_event_id == 0:
        print(f"Required event IDs (769, 770) not found in the data for file {file_path}.")
        return

    epochs = mne.Epochs(raw, events, event_id={'left_hand': left_hand_event_id, 'right_hand': right_hand_event_id},
                        tmin=-0.2, tmax=0.8, baseline=None)
    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1] - left_hand_event_id  # Adjust to match event IDs

    # Extract features using combined CSP and PCA
    features = extract_features_csp_pca(epochs_data, labels)

    # Load the trained models
    svm = load('svm_model.joblib')
    lda = load('lda_model.joblib')

    # Make predictions
    svm_predictions = svm.predict(features)
    lda_predictions = lda.predict(features)

    # Evaluate the predictions
    svm_evaluation_score = accuracy_score(labels, svm_predictions)
    lda_evaluation_score = accuracy_score(labels, lda_predictions)

    print(f'SVM Evaluation Accuracy: {svm_evaluation_score}')
    print(f'LDA Evaluation Accuracy: {lda_evaluation_score}')

if __name__ == '__main__':
    file_path = "data/A04E.gdf"
    evaluate_model(file_path, apply_ica_flag=True)
