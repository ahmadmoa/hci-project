import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from mne.decoding import CSP

def extract_features(epochs, labels):
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    features = csp.fit_transform(epochs, labels)
    return features

def train_svm(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    score = svm.score(X_test, y_test)
    return svm, score

def train_lda(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    lda = LDA()
    lda.fit(X_train, y_train)
    score = lda.score(X_test, y_test)
    return lda, score