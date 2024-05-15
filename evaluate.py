import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score
from mne.decoding import CSP

def evaluate_model():
    # Load the saved test data
    test_features = np.load('X_test_features.npy')
    y_test = np.load('y_test.npy')

    # Load the trained models
    svm = load('svm_model.joblib')
    lda = load('lda_model.joblib')
    lr = load('lr_model.joblib')

    # Make predictions
    svm_predictions = svm.predict(test_features)
    lda_predictions = lda.predict(test_features)
    lr_predictions = lr.predict(test_features)

    # Evaluate the predictions
    svm_evaluation_score = accuracy_score(y_test, svm_predictions)
    lda_evaluation_score = accuracy_score(y_test, lda_predictions)
    lr_evaluation_score = accuracy_score(y_test, lr_predictions)

    print(f'SVM Evaluation Accuracy: {svm_evaluation_score}')
    print(f'LDA Evaluation Accuracy: {lda_evaluation_score}')
    print(f'LR Evaluation Accuracy: {lr_evaluation_score}')

if __name__ == '__main__':
    evaluate_model()
