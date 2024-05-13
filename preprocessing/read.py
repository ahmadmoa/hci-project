import mne

def load_data(file_path):
    raw = mne.io.read_raw_gdf(file_path, preload=True)
    return raw

