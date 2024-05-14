import numpy as np
from mne.preprocessing import ICA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from mne.decoding import CSP
from sklearn.decomposition import PCA


def apply_ica(raw, eog_channels=None):
    ica = ICA(n_components=15, random_state=97, max_iter=800)
    ica.fit(raw)

    if eog_channels:
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels)
    else:
        eog_indices, eog_scores = ica.find_bads_eog(raw)

    ica.exclude = eog_indices
    raw_cleaned = ica.apply(raw)
    return raw_cleaned

def extract_features_csp_pca(epochs, labels):
    # Apply CSP
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    csp_features = csp.fit_transform(epochs, labels)

    # Apply PCA
    pca = PCA(n_components=4)
    pca_features = pca.fit_transform(csp_features)

    return pca_features


def extract_features(epochs, labels):
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    features = csp.fit_transform(epochs, labels)
    return features
def extract_features_pca(epochs):
    pca = PCA(n_components=4)
    features = pca.fit_transform(epochs.reshape(len(epochs), -1))
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