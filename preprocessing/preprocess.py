
def preprocess_data(raw):
    raw.filter(8., 30., fir_design='firwin')
    return raw

# Usage example:
# raw = load_data('data/A01T.gdf')
# raw = preprocess_data(raw)
