import numpy as np
import pandas as pd
import torch
from pathlib import Path
from lit_saint import Saint, SaintConfig, SaintDatamodule, SaintTrainer
from pytorch_lightning import Trainer
from typing import Any

from utils import verify_and_align_submission
from make_dataset import make_dataset
from config import *


def load_model():
    print("Loading trained SAINT models from directory")
    airport_models = {}
    for airport in airports:
        checkpoint = torch.load(model_directory / f'saint_{airport}_{VERSION}.pt')
        
        # Initialize model with saved configuration
        model = Saint(
            categories=checkpoint['categorical_dims'],
            continuous=checkpoint['numerical_columns'],
            config=checkpoint['config'],
            dim_target=1
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        airport_models[airport] = {
            'model': model,
            'categorical_dims': checkpoint['categorical_dims'],
            'numerical_columns': checkpoint['numerical_columns'],
            'scaler': checkpoint.get('scaler', None)  # Get scaler if it exists
        }
    
    return airport_models

def prepare_inference_data(features, num_features, cat_features):
    """Prepare dataset for SAINT model inference"""
    dropped_columns = set(['timestamp', 'arrivals', 'interval_start', 'tfm_arrivals_diff_diff', 'ID', 
                         'wind_speed', 'wind_direction', 'tbfm_var_arrival_error'])
    
    data = features.copy()
    data.drop(columns=[col for col in dropped_columns if col in data.columns], inplace=True)
    
    # Handle categorical features
    for feature in cat_features:
        if feature in data.columns:
            data[feature] = data[feature].astype(str)
            data[feature] = data[feature].replace('nan', 'SAINT_NAN')
            data[feature] = pd.Categorical(data[feature])
    
    # Handle numerical features
    for feature in num_features:
        if feature in data.columns:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
    
    return data

def predict(models, airport, features):
    """Make predictions for a set of flights at a single airport and prediction time."""
    model_info = models[airport]
    model = model_info['model']
    
    # Prepare inference data
    data = prepare_inference_data(
        features,
        model_info['numerical_columns'],
        [col for col in features.columns if col not in model_info['numerical_columns'] 
         and col not in ['ID', 'arrivals']]
    )
    
    # Label data as test
    data['split'] = 'train'
    
    # Create datamodule for inference
    data_module = SaintDatamodule(
        df=data,
        target='arrivals',
        split_column='split'
    )
    
    # Replace the datamodule's scaler with the saved one
    data_module.scaler = model_info['scaler']
    
    # Setup trainer and saint_trainer for prediction
    trainer = Trainer()
    pretrainer = Trainer()
    saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
    
    # Get predictions
    with torch.no_grad():
        predictions = saint_trainer.predict(model=model, datamodule=data_module, df=data)
        predictions = predictions['prediction']  # Extract predictions from dictionary
    
    # Add predictions to features DataFrame
    features['Value'] = predictions.flatten()
    submission = features[['ID', 'Value']]
    
    # Save airport-specific predictions
    submission.to_csv(prediction_directory / f'{airport}_predictions_{VERSION}.csv', index=False)
    
    return submission


if __name__ == "__main__":
    # Generate predictions for submission format using provided competition processing pipeline
    prediction_directory.mkdir(parents=True, exist_ok=True)
    models = load_model()
    submissions = []
    
    for airport in airports:
        test_df = make_dataset(airport, split='test')
        submissions.append(predict(models, airport, test_df))
    
    full_df = pd.concat(submissions, ignore_index=True)
    full_df['Value'].clip(lower=0, inplace=True)
    
    aligned_df = verify_and_align_submission(full_df)
    
    if aligned_df is not None:
        submission_path = prediction_directory / f'submission_{VERSION}.csv'
        aligned_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {str(submission_path)}")