#%%
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from config import *

airports = ['KATL', 'KCLT', 'KDEN', 'KDFW', 'KJFK', 'KMEM', 'KMIA', 'KORD', 'KPHX', 'KSEA']
VERSION = 'v4'

results = {}

for airport in airports:
    print(f"\nProcessing {airport}...")
    
    hyper_params = {
        'iterations': 9000, 
        'learning_rate': 0.02,
        'l2_leaf_reg': train_params[airport]['l2_leaf_reg'],
        'max_depth': train_params[airport]['depth'],
        'loss_function': 'RMSE',
        'thread_count': -1,
        'metric_period': 50,
        'eval_metric': 'RMSE',
    }
    
    model = CatBoostRegressor(**hyper_params, random_seed=0)
    
    # Load data for current airport
    data_df = pd.read_pickle(f'/home/brianhu/workspace/throughput-challenge/data/train_data/{airport}_train_features_v4_h0.pkl.zip')
    
    dropped_columns = set(['timestamp', 'arrivals', 'interval_start', 'ID'])
    X = data_df.drop(columns=dropped_columns.intersection(data_df.columns))
    y = data_df['arrivals']
    
    # Get categorical columns
    categorical_columns = []
    for col in X.columns:
        if data_df[col].dtype == 'object' or data_df[col].dtype == 'category':
            categorical_columns.append(col)

    # Preprocess categorical features
    for col in categorical_columns:
        if col in X.columns:
            X[col] = X[col].fillna(f"Missing {col}")
            X[col] = X[col].astype(str)

    # Split data
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25, random_state=21)
    train_pool = Pool(X_train, y, cat_features=categorical_columns)
    eval_pool = Pool(X_eval, y_eval, cat_features=categorical_columns)

    # Feature selection
    result = model.select_features(
        X=train_pool,
        eval_set=eval_pool,
        features_for_select=f'0-{len(X.columns)-1}',
        num_features_to_select=24,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=False,
        logging_level='Silent',
        plot=True,
        steps=8,
    )

    # Process results
    loss_values = result['loss_graph']['loss_values']
    min_loss_idx = np.argmin(loss_values)
    eliminated_features = result['eliminated_features_names'][:min_loss_idx]
    print(airport, eliminated_features)

    # Store results for this airport
    results[airport] = {
        'min_loss': float(min(loss_values)),  # Convert numpy float to native Python float
        'eliminated_features': eliminated_features,
        'eliminated_features_count': len(eliminated_features)
    }

# Save all results to JSON file
with open('eliminated_features_all_airports.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nProcessing complete. Results saved to 'eliminated_features_all_airports.json'")
# %%