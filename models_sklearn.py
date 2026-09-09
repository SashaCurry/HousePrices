from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from config import config
from data_handle import *


def train_model_sklearn(train_data, model_name='linear_regression'):
    train_data_handled = handling_for_linear(train_data)

    X = train_data_handled.drop(columns=['SalePrice'])
    y = train_data_handled['SalePrice']

    base_pipeline = None
    if model_name == 'linear_regression':
        base_pipeline = Pipeline([
            ('scale', preprocessor),
            ('model', SGDRegressor(**config.linreg.params))
        ])
    elif model_name == 'linear_regression_l1':
        base_pipeline = Pipeline([
            ('scale', preprocessor),
            ('model', SGDRegressor(**config.linreg_l1.params))
        ])
    elif model_name == 'linear_regression_l2':
        base_pipeline = Pipeline([
            ('scale', preprocessor),
            ('model', SGDRegressor(**config.linreg_l2.params))
        ])
    elif model_name == 'linear_regression_elasticnet':
        base_pipeline = Pipeline([
            ('scale', preprocessor),
            ('model', SGDRegressor(**config.linreg_elnet.params))
        ])
    elif model_name == 'knn':
        base_pipeline = Pipeline([
            ('scale', preprocessor),
            ('model', KNeighborsRegressor(**config.knn.params))
        ])
    elif model_name == 'decision_tree':
        base_pipeline = Pipeline([
            ('scale', preprocessor),
            ('model', DecisionTreeRegressor(**config.decision_tree.params))
        ])
    elif model_name == 'random_forest':
        base_pipeline = Pipeline([
            ('scale', preprocessor),
            ('model', RandomForestRegressor(**config.random_forest.params))
        ])

    model = TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    scores = cross_val_score(model, X, y, cv=config.training.n_splits, scoring='neg_root_mean_squared_log_error')

    model_acc = round(-scores.mean(), 2)
    model_std = round(scores.std(), 2)

    base_pipeline.fit(X, y)

    return base_pipeline, model_acc, model_std


def test_model_sklearn(data, model, model_name):
    X_test = handling_for_linear(data)

    preds = model.predict(X_test)

    df = pd.DataFrame({'Id': data['Id'],
                       'SalePrice': preds})
    df.to_csv(path_or_buf=f'{config.paths.path_save_csv}{model_name}_preds.csv',
              index=False)