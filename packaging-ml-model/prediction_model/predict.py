import pandas as pd
import numpy as np
import joblib # handle model serialization & deserialization

from pathlib import Path
import os
import sys

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.processing.data_handling import load_dataset, load_pipeline
from prediction_model.config import config

classification_pipeline = load_pipeline(config.MODEL_NAME)

# for test_data use
def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    pred = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(pred==1,'Y','N')
    result = {"prediction":output}
    return result

# def generate_prediction():
#     test_data = load_dataset(config.TESTFILE)
#     prediction = classification_pipeline.predict(test_data[config.FEATURES])
#     output = np.where(prediction == 1, 'Y', 'N')
#     return output

if __name__ == '__main__':
    generate_prediction()