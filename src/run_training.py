import json
import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from config import *


def train_new_model(data_df):
    print(f'Training model for {airport}')

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
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # train_pool = Pool(X_train, y_train, cat_features=categorical_columns)
    # eval_pool = Pool(X_test, y_test, cat_features=categorical_columns)
    
    train_pool = Pool(X, y, cat_features=categorical_columns)

    hyper_params = {
        'iterations': train_params[airport]['iterations'],
        'learning_rate': 0.01,
        'l2_leaf_reg': train_params[airport]['l2_leaf_reg'],
        'max_depth': train_params[airport]['depth'],
        'loss_function': 'RMSE',
        'thread_count': -1,
        'metric_period': 50,
        'eval_metric': 'RMSE',
        'random_strength': 1,
    }

    model = CatBoostRegressor(**hyper_params)
    model.fit(
        train_pool, 
        use_best_model=True,
        # eval_set=eval_pool,
        # early_stopping_rounds=200,
        verbose=True)
    model.save_model(model_directory / f'catboost_{airport}_{VERSION}.cbm')

    return model, model.get_best_score()


if __name__ == "__main__":
    # Train catboost model on each airport
    metrics = defaultdict(list)
    model_directory.mkdir(parents=True, exist_ok=True)
    
    for airport in airports:
        for i in range(1):
            data_df = pd.read_pickle(train_directory / f'{airport}_train_features_{VERSION}_h{i}.pkl.zip')
            model, score = train_new_model(data_df)
        print(score)
        
        for key in score:
            metrics[key].append(score[key]['RMSE'])
    
    with open(model_directory / f'catboost_metrics_{VERSION}.log', 'a') as file:
        file.write(json.dumps(metrics) + '\n')