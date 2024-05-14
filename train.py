from preprocessing.preprocessing import load_and_preprocess
from models.model import apply_ica, extract_features_csp_pca
import mne
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.utils import resample
from joblib import dump


def train_and_save_model(file_path, apply_ica_flag=True, augment_data=True):
    raw = load_and_preprocess(file_path)

    if apply_ica_flag:
        eog_channels = ['EOG-left', 'EOG-central', 'EOG-right']
        raw = apply_ica(raw, eog_channels=eog_channels)

    events, event_id = mne.events_from_annotations(raw)
    print("Events Array:", events)
    print("Event ID Dictionary:", event_id)

    # Ensure that the events are correctly accessed using internal indices
    if 7 not in event_id.values() or 8 not in event_id.values():
        print(f"Required event IDs (769, 770) not found in the data for file {file_path}.")
        return

    # Correctly accessing the event ID mappings
    epochs = mne.Epochs(raw, events, event_id={'left_hand': event_id['769'], 'right_hand': event_id['770']},
                        tmin=-0.2, tmax=0.8, baseline=None)
    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1] - event_id['769']

    if augment_data:
        augmented_data = []
        augmented_labels = []
        for _ in range(5):
            aug_data, aug_labels = resample(epochs_data, labels)
            augmented_data.append(aug_data)
            augmented_labels.append(aug_labels)

        augmented_data = np.concatenate(augmented_data, axis=0)
        augmented_labels = np.concatenate(augmented_labels, axis=0)

        epochs_data = np.concatenate([epochs_data, augmented_data], axis=0)
        labels = np.concatenate([labels, augmented_labels], axis=0)

    features = extract_features_csp_pca(epochs_data, labels)

    # Grid search and training for SVM
    svm_param_grid = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}
    svm = SVC()
    svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, scoring='accuracy')
    svm_grid_search.fit(features, labels)
    dump(svm_grid_search.best_estimator_, 'svm_model.joblib')

    print(f'Best SVM Parameters: {svm_grid_search.best_params_}')
    print(f'Best SVM Cross-Validation Accuracy: {svm_grid_search.best_score_}')

    # Grid search and training for LDA
    lda_param_grid = [{'solver': ['svd']}, {'solver': ['lsqr', 'eigen'], 'shrinkage': ['auto', 0.1, 0.5, 1.0]}]
    lda = LDA()
    lda_grid_search = GridSearchCV(lda, lda_param_grid, cv=5, scoring='accuracy')
    lda_grid_search.fit(features, labels)
    dump(lda_grid_search.best_estimator_, 'lda_model.joblib')

    print(f'Best LDA Parameters: {lda_grid_search.best_params_}')
    print(f'Best LDA Cross-Validation Accuracy: {lda_grid_search.best_score_}')

if __name__ == '__main__':
    file_path = "data/A01T.gdf"
    train_and_save_model(file_path, apply_ica_flag=True, augment_data=True)
