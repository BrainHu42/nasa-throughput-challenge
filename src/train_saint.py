import torch

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from lit_saint import Saint, SaintConfig, SaintDatamodule, SaintTrainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, seed_everything
from config import *

torch.set_float32_matmul_precision('high')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class RMSECallback(Callback):
    def __init__(self):
        self.train_rmse = []
        self.val_rmse = []
        self.best_val_rmse = float('inf')
        
    def on_train_epoch_end(self, trainer, pl_module):
        # Get the raw MSE loss
        train_mse = trainer.callback_metrics.get('train_loss', None)
        if train_mse is not None:
            # Convert MSE to RMSE
            train_rmse = torch.sqrt(train_mse)
            self.train_rmse.append(float(train_rmse))
            
    def on_validation_epoch_end(self, trainer, pl_module):
        val_mse = trainer.callback_metrics.get('val_loss', None)
        if val_mse is not None:
            # Convert MSE to RMSE
            val_rmse = torch.sqrt(val_mse)
            self.val_rmse.append(float(val_rmse))
            # Update best RMSE
            if float(val_rmse) < self.best_val_rmse:
                self.best_val_rmse = float(val_rmse)
    
# Configure SAINT model
cfg = SaintConfig()
seed_everything(21, workers=True)

train_params = {
    'KATL': {'iterations': 8700},
    'KCLT': {'iterations': 7800},
    'KDEN': {'iterations': 9100}, 
    'KDFW': {'iterations': 9100},
    'KJFK': {'iterations': 6700},
    'KMEM': {'iterations': 8600},
    'KMIA': {'iterations': 6200},
    'KORD': {'iterations': 10400},
    'KPHX': {'iterations': 6500},
    'KSEA': {'iterations': 6400},
}

def prepare_dataset(data_df):
    """Prepare dataset for SAINT model training"""
    # Define numerical and categorical features based on the dataset
    num_features = [
        'arrivals_variance_h1', 'avg_arrival_error', 'var_arrival_error',
        'tbfm_avg_arrival_error', 'avg_time_between_arrivals',
        'var_time_between_arrivals', 'tbfm_avg_time_between_estimated_arrivals',
        'hour_cos', 'hour_sin', 'day_cos', 'day_sin', 'month_cos', 'month_sin'
    ]
    
    # Drop columns that won't be used as features
    dropped_columns = ['timestamp', 'interval_start', 'tfm_arrivals_diff_diff', 'ID', 
                      'wind_speed', 'wind_direction', 'tbfm_var_arrival_error']
    
    data = data_df.copy()
    data.drop(columns=[col for col in dropped_columns if col in data.columns], inplace=True)
    
    # Identify categorical features
    cat_features = [col for col in data.columns if col not in num_features and col != 'arrivals']
    
    # Handle categorical features
    for feature in cat_features:
        # Convert to string first to ensure proper category handling
        data[feature] = data[feature].astype(str)
        # Add 'SAINT_NAN' to categories
        unique_vals = data[feature].unique().tolist()
        if 'nan' in unique_vals:
            unique_vals.remove('nan')
        unique_vals.append('SAINT_NAN')
        # Create categorical type with 'SAINT_NAN' as a possible category
        data[feature] = pd.Categorical(
            data[feature].replace('nan', 'SAINT_NAN'),
            categories=unique_vals
        )
    
    return data, num_features, cat_features



def train_saint_model(airport, data_df):
    """Train SAINT model for a specific airport"""
    print(f'Training model for {airport}')
    
    # Prepare dataset
    data, num_features, cat_features = prepare_dataset(data_df)
    
    # Split data into train and validation sets
    df_train, df_val = train_test_split(data, test_size=0.1, random_state=42)
    df_train["split"] = "train"
    df_val["split"] = "validation"
    df = pd.concat([df_train, df_val])
    
    # Initialize data module and model
    data_module = SaintDatamodule(
        df=df, 
        target="arrivals", 
        split_column="split"
    )
    data_module.data_loader_params['num_workers'] = 4
    
    model = Saint(
        categories=data_module.categorical_dims,
        continuous=data_module.numerical_columns,
        config=cfg,
        dim_target=1
    )
    
    # Setup callbacks
    rmse_callback = RMSECallback()

    # Setup trainers with callbacks
    pretrainer = Trainer(
        max_epochs=cfg.pretrain.epochs,
        callbacks=[rmse_callback],
        accelerator='gpu',
        devices=1,
        precision="bf16",  # Add this for mixed precision training
    )
    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        callbacks=[rmse_callback],
        accelerator='gpu',  # Add this
        devices=1,          # Add this
        precision="bf16",  # Add this for mixed precision training
    )
    saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
    
    saint_trainer.fit(model=model, datamodule=data_module, enable_pretraining=True)
    
    return model, data_module, rmse_callback.best_val_rmse


if __name__ == "__main__":
    metrics = defaultdict(list)
    
    for airport in airports:
        # Load dataset
        data_df = pd.read_pickle(train_directory / f'{airport}_train_features_{VERSION}.pkl.zip')
        
        # Train model and get metrics
        model, data_module, best_rmse = train_saint_model(airport, data_df)
        metrics['RMSE'].append(best_rmse)
        print(f"Best RMSE for {airport}: {best_rmse}")
        
        # Save model
        model_path = model_directory / f'saint_{airport}_{VERSION}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': cfg,
            'categorical_dims': data_module.categorical_dims,
            'numerical_columns': data_module.numerical_columns
        }, model_path)
    
    # Save metrics
    with open(model_directory / f'saint_metrics_{VERSION}.log', 'a') as file:
        file.write(json.dumps(metrics) + '\n')