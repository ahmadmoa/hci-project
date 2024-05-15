from preprocessing.preprocessing import load_and_preprocess
from models.model import apply_ica
import mne
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from joblib import dump
from mne.decoding import CSP

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

    # Split the data into 75% training and 25% testing
    X_train, X_test, y_train, y_test = train_test_split(epochs_data, labels, test_size=0.25, random_state=42)

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

    # Feature extraction using CSP
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    train_features = csp.fit_transform(X_train, y_train)
    test_features = csp.transform(X_test)

    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



    # Grid search and training for LDA
    lda_param_grid = [{'solver': ['svd']}, {'solver': ['lsqr', 'eigen'], 'shrinkage': ['auto', 0.1, 0.5, 1.0]}]
    lda = LDA()
    lda_grid_search = GridSearchCV(lda, lda_param_grid, cv=cv, scoring='accuracy')
    lda_grid_search.fit(train_features, y_train)
    dump(lda_grid_search.best_estimator_, 'lda_model.joblib')

    print(f'Best LDA Parameters: {lda_grid_search.best_params_}')
    print(f'Best LDA Cross-Validation Accuracy: {lda_grid_search.best_score_}')

    # Grid search and training for Logistic Regression
    lr_param_grid = [
        {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']},
        {'C': [0.01, 0.1, 1, 10], 'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0.1, 0.5, 0.7]}
    ]
    lr = LogisticRegression(max_iter=1000)
    lr_grid_search = GridSearchCV(lr, lr_param_grid, cv=cv, scoring='accuracy')
    lr_grid_search.fit(train_features, y_train)
    dump(lr_grid_search.best_estimator_, 'lr_model.joblib')

    print(f'Best LR Parameters: {lr_grid_search.best_params_}')
    print(f'Best LR Cross-Validation Accuracy: {lr_grid_search.best_score_}')

    # Grid search and training for SVM
    svm_param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    svm = SVC()
    svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=cv, scoring='accuracy')
    svm_grid_search.fit(train_features, y_train)
    dump(svm_grid_search.best_estimator_, 'svm_model.joblib')

    print(f'Best SVM Parameters: {svm_grid_search.best_params_}')
    print(f'Best SVM Cross-Validation Accuracy: {svm_grid_search.best_score_}')

    # Save the test data, feature extraction model (CSP), and training features for evaluation
    np.save('X_test_features.npy', test_features)
    np.save('y_test.npy', y_test)
    dump(csp, 'csp_model.joblib')

if __name__ == '__main__':
    file_path = "data/old data/A01T.gdf"
    train_and_save_model(file_path, apply_ica_flag=True, augment_data=True)
