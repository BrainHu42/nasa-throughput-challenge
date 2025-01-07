import pandas as pd
import json
from catboost import CatBoostRegressor, Pool
from config import *
from itertools import product
from sklearn.model_selection import train_test_split


def custom_grid_search(train_pool, test_pool, param_grid):
    # Generate all combinations of parameters
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    results = []
    
    best_score = float('inf')
    best_params = None
    
    for params in param_combinations:
        print(f"\nTrying parameters: {params}")
        
        # Fixed parameters
        model = CatBoostRegressor(
            loss_function='RMSE',
            thread_count=-1,
            eval_metric='RMSE',
            metric_period=100,
            **params  # Add grid parameters
        )
        
        # Train with test set for evaluation
        model.fit(
            train_pool,
            eval_set=test_pool,
            early_stopping_rounds=300,
            verbose=True,
            use_best_model=True
        )
        
        # Get the best validation score
        eval_score = model.get_best_score()['validation']['RMSE']
        
        result = {
            'params': params,
            'validation_score': eval_score
        }
        results.append(result)
        
        # Track best parameters
        if eval_score < best_score:
            best_score = eval_score
            best_params = params
    
    return {
        'params': best_params,
        'cv_results': results,
        'best_score': best_score
    }


def perform_grid_search(data_df, airport):
    print(f'Performing grid search for {airport}')
    dropped_columns = set(['timestamp', 'arrivals', 'interval_start', 'ID', '4hour_group']).union(train_params[airport]['dropped_features'])
    X = data_df.drop(columns=dropped_columns.intersection(data_df.columns))
    y = data_df['arrivals']
    print(X.columns)
    
    # Get categorical columns
    categorical_columns = []
    for col in X.columns:
        if data_df[col].dtype == 'object' or data_df[col].dtype == 'category':
            categorical_columns.append(col)
            
    for col in categorical_columns:
        if col in X.columns:
            X[col] = X[col].fillna(f"Missing {col}")
            X[col] = X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_pool = Pool(X_train, y_train, cat_features=categorical_columns)
    eval_pool = Pool(X_test, y_test, cat_features=categorical_columns)

    # Define the grid of hyperparameters to search
    grid = {
        'iterations': [8000],
        'learning_rate': [0.03],
        'max_depth': [7, 8, 9],
        'l2_leaf_reg': [5, 7, 10],
    }

    # Perform custom grid search with test set
    grid_search_result = custom_grid_search(train_pool, eval_pool, grid)
    
    return grid_search_result


if __name__ == "__main__":
    # Perform grid search for each airport
    for airport in airports:
        # Extract and save dataset
        # data_df = make_dataset(airport)
        data_df = pd.read_pickle(train_directory / f'{airport}_train_features_{VERSION}_h0.pkl.zip')
        
        # Perform grid search
        grid_search_result = perform_grid_search(data_df, airport)
        
        print(grid_search_result['params'])
        
        # Save grid search result as JSON
        result_file = model_directory / f'grid_search_result_{airport}.json'
        
        # Convert numpy types to Python native types for JSON serialization
        serializable_result = json.loads(json.dumps(grid_search_result, default=str))
        
        with open(result_file, 'w') as f:
            json.dump(serializable_result, f, indent=4)
        
        print(f"Grid search result for {airport} saved to {result_file}")
