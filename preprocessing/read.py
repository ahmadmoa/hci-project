import mne

def load_data(file_path):
    raw = mne.io.read_raw_gdf(file_path, preload=True)
    return raw

def visualize_data(raw):
    # Plot the raw data
    raw.plot()

    # Plot the power spectral density (PSD)
    raw.plot_psd()

    # Plot events (if available)
    try:
        events = mne.find_events(raw, stim_channel='STI 014')
    except ValueError:
        events, _ = mne.events_from_annotations(raw)
    mne.viz.plot_events(events, raw.info['sfreq'])

    # Plot the topography of the data
    raw.plot_sensors()

    # Plot the channel locations
    raw.plot_sensors(kind='3d')

    # Plot the average reference
    raw.plot(proj=True)

# Load the data
raw_data = load_data("../data/A01T.gdf")

# Visualize the data
visualize_data(raw_data)
