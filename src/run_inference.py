import numpy as np
import pandas as pd

from catboost import Pool, CatBoostRegressor
from typing import Any

from utils import verify_and_align_submission
from make_dataset import make_dataset
from config import *

def load_model():
    print("Loading trained model from directory")
    airport_models = {}
    for airport in airports:
        airport_models[airport] = CatBoostRegressor().load_model(model_directory / f'catboost_{airport}_{VERSION}.cbm')
        
    return airport_models


def predict(model, airport, features):
    """Make predictions for the a set of flights at a single airport and prediction time.

    Returns:
        pd.DataFrame: Predictions for all of the flights in the partial_submission_format
    """
    dropped_columns = set(['timestamp', 'arrivals', 'interval_start', 'ID']).union(train_params[airport]['dropped_features'])
    X_test = features.drop(columns=dropped_columns.intersection(features.columns))
    print(X_test.columns)
    
    # Get categorical columns
    categorical_columns = []
    for col in X_test.columns:
        if X_test[col].dtype == 'object' or X_test[col].dtype == 'category':
            categorical_columns.append(col)
    
    # Preprocess categorical features - fill NaN values with "Missing"
    for col in categorical_columns:
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(f"Missing {col}")
            # Ensure all values are strings
            X_test[col] = X_test[col].astype(str)
    
    features['Value'] = model[airport].predict(X_test)
    submission = features[['ID', 'Value']]
    submission.to_csv(prediction_directory / f'{airport}_predictions_{VERSION}.csv', index=False)
    
    return submission


import multiprocessing as mp
from functools import partial
    
if __name__ == "__main__":
    # Process datasets in parallel
    pool = mp.Pool(processes=5)
    make_dataset_partial = partial(make_dataset, split='test')
    pool.map(make_dataset_partial, airports)
    
    pool.close()
    pool.join()
    
    # Generate predictions for submission format using provided competition processing pipeline
    prediction_directory.mkdir(parents=True, exist_ok=True)
    model = load_model()
    submissions = []
    for airport in airports:
        # Read saved extracted features
        test_df = pd.read_pickle(train_directory / f'{airport}_test_features_{VERSION}_h0.pkl.zip')
        submissions.append(predict(model, airport, test_df))
    
    full_df = pd.concat(submissions, ignore_index=True)
    full_df['Value'] = full_df['Value'].clip(lower=0).round()
    
    aligned_df = verify_and_align_submission(full_df)
    
    if aligned_df is not None:
        submission_path = prediction_directory / f'submission_{VERSION}.csv'
        aligned_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {str(submission_path)}")

