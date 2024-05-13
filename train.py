from preprocessing.preprocessing import preprocess_data, load_and_preprocess
from preprocessing.read import load_data
from models.modle import extract_features, train_svm, train_lda
import mne
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.utils import resample



if __name__ == '__main__':
    train_files = ["data/A01T.gdf"]

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

    # Data augmentation (optional)
    augmented_data = []
    augmented_labels = []

    for _ in range(5):  # Example: augmenting data 5 times
        aug_data, aug_labels = resample(all_epochs_data, all_labels)
        augmented_data.append(aug_data)
        augmented_labels.append(aug_labels)

    augmented_data = np.concatenate(augmented_data, axis=0)
    augmented_labels = np.concatenate(augmented_labels, axis=0)

    # Combine original and augmented data
    combined_data = np.concatenate([all_epochs_data, augmented_data], axis=0)
    combined_labels = np.concatenate([all_labels, augmented_labels], axis=0)

    features = extract_features(combined_data, combined_labels)

    # Grid search for SVM
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svm = SVC()
    svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, scoring='accuracy')
    svm_grid_search.fit(features, combined_labels)
    best_svm = svm_grid_search.best_estimator_
    best_svm_score = svm_grid_search.best_score_

    print(f'Best SVM Parameters: {svm_grid_search.best_params_}')
    print(f'Best SVM Cross-Validation Accuracy: {best_svm_score}')

    # Corrected grid search for LDA
    lda_param_grid = [
        {'solver': ['svd']},
        {'solver': ['lsqr', 'eigen'], 'shrinkage': ['auto', 0.1, 0.5, 1.0]}
    ]
    lda = LDA()
    lda_grid_search = GridSearchCV(lda, lda_param_grid, cv=5, scoring='accuracy')
    lda_grid_search.fit(features, combined_labels)
    best_lda = lda_grid_search.best_estimator_
    best_lda_score = lda_grid_search.best_score_

    print(f'Best LDA Parameters: {lda_grid_search.best_params_}')
    print(f'Best LDA Cross-Validation Accuracy: {best_lda_score}')

    # Save the trained models for later use
    from joblib import dump
    dump(best_svm, 'svm_model.joblib')
    dump(best_lda, 'lda_model.joblib')
