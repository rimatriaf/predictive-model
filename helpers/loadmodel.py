from tensorflow.keras.models import load_model
import os

_model = None

def get_model():
    global _model
    if _model is None:
        try:
            print("⏳ Loading LSTM model...")

            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "..", "model", "lstm.h5")
            model_path = os.path.abspath(model_path)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            _model = load_model(model_path, compile=False)
            print("LSTM Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e

    return _model
