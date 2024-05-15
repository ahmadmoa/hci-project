from preprocessing.preprocessing import load_and_preprocess
from models.model import apply_ica, extract_features_csp_pca
import mne
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
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

    if '769' not in event_id or '770' not in event_id:
        print(f"Required event IDs (769, 770) not found in the data for file {file_path}.")
        return

    epochs = mne.Epochs(raw, events, event_id={'left_hand': event_id['769'], 'right_hand': event_id['770']},
                        tmin=-0.2, tmax=0.8, baseline=None)
    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1] - event_id['769']

    X_train, X_test, y_train, y_test = train_test_split(epochs_data, labels, test_size=0.2, random_state=42)

    if augment_data:
        augmented_data = []
        augmented_labels = []
        for _ in range(5):
            aug_data, aug_labels = resample(X_train, y_train)
            augmented_data.append(aug_data)
            augmented_labels.append(aug_labels)

        augmented_data = np.concatenate(augmented_data, axis=0)
        augmented_labels = np.concatenate(augmented_labels, axis=0)

        X_train = np.concatenate([X_train, augmented_data], axis=0)
        y_train = np.concatenate([y_train, augmented_labels], axis=0)

    train_features = extract_features_csp_pca(X_train, y_train)

    # Grid search and training for SVM
    svm_param_grid = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}
    svm = SVC()
    svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, scoring='accuracy')
    svm_grid_search.fit(train_features, y_train)
    dump(svm_grid_search.best_estimator_, 'svm_model.joblib')

    print(f'Best SVM Parameters: {svm_grid_search.best_params_}')
    print(f'Best SVM Cross-Validation Accuracy: {svm_grid_search.best_score_}')

    # Grid search and training for LDA
    lda_param_grid = [{'solver': ['svd']}, {'solver': ['lsqr', 'eigen'], 'shrinkage': ['auto', 0.1, 0.5, 1.0]}]
    lda = LDA()
    lda_grid_search = GridSearchCV(lda, lda_param_grid, cv=5, scoring='accuracy')
    lda_grid_search.fit(train_features, y_train)
    dump(lda_grid_search.best_estimator_, 'lda_model.joblib')

    print(f'Best LDA Parameters: {lda_grid_search.best_params_}')
    print(f'Best LDA Cross-Validation Accuracy: {lda_grid_search.best_score_}')

    # Grid search and training for Logistic Regression
    lr_param_grid = {'C': [0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2', 'elasticnet', 'none'], 'solver': ['liblinear', 'saga']}
    lr = LogisticRegression(max_iter=1000)
    lr_grid_search = GridSearchCV(lr, lr_param_grid, cv=5, scoring='accuracy')
    lr_grid_search.fit(train_features, y_train)
    dump(lr_grid_search.best_estimator_, 'lr_model.joblib')

    print(f'Best LR Parameters: {lr_grid_search.best_params_}')
    print(f'Best LR Cross-Validation Accuracy: {lr_grid_search.best_score_}')

    # Save the test data for evaluation
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

if __name__ == '__main__':
    file_path = "data/old data/A01T.gdf"
    train_and_save_model(file_path, apply_ica_flag=True, augment_data=True)
