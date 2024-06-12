import pandas as pd
import numpy as np

from pathlib import Path
import os
import sys

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.processing.data_handling import load_dataset, save_pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
import prediction_model.pipeline as pipe


def perform_training():
    # Load data
    train_data = load_dataset(config.TRAINFILE)
    
    train_y = train_data[config.TARGET].map({'Y': 1, 'N': 0})
    pipe.classification_pipeline.fit(train_data[config.FEATURES], train_y)
    save_pipeline(pipe.classification_pipeline)
    
if __name__ == '__main__':
    perform_training()