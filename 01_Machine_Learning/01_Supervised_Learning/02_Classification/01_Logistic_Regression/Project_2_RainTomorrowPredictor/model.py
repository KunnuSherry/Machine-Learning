import joblib
import pickle

# Load the joblib file
try:
    aussie_rain = joblib.load('aussie_rain.joblib')
    print("File loaded successfully:", type(aussie_rain))
    print("Keys in aussie_rain:", aussie_rain.keys())  # Check available keys
except Exception as e:
    print("Error loading file:", e)
    exit()  # Stop execution if file loading fails

# Check if 'model' exists
if 'model' in aussie_rain:
    model = aussie_rain['model']
    print("Model key exists:", type(model))
else:
    print("Key 'model' not found in aussie_rain")
    exit()

pickle.dump(model, open('iri.pkl', 'wb'))