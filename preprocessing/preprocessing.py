
import mne

def load_data(file_path):
    raw = mne.io.read_raw_gdf(file_path, preload=True)
    return raw

def preprocess_data(raw):
    raw.filter(8., 30., fir_design='firwin')
    return raw

def load_and_preprocess(file_path):
    raw = load_data(file_path)
    raw = preprocess_data(raw)
    return raw
# Usage example:
# raw = load_data('data/A01T.gdf')
# raw = preprocess_data(raw)
