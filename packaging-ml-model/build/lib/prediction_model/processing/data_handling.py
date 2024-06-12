import os
import pandas as pd
import joblib # handle model serialization & deserialization

from pathlib import Path
import os
import sys
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))
from prediction_model.config import config






# Load dataset
def load_dataset(file_name):
    _data = pd.read_csv(os.path.join(config.DATAPATH, file_name))
    return _data

# Serialization
def save_pipeline(pipeline_to_persist):
    model_save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_persist, model_save_path)
    print(f"saved pipeline: {config.MODEL_NAME}")

# Deserialization
def load_pipeline(pipeline_to_load):
    model_save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(model_save_path)
    print(f"Model loaded")
    return model_loaded
    