from sklearn.ensemble import BaggingRegressor, StackingRegressor
from sklearn.model_selection import cross_val_score
import importlib

from config import config
from data_handle import *

def train_bagging(train_data):
    train_data_handled = handling_for_linear(train_data)

    X = train_data_handled.drop(columns='SalePrice')
    y = train_data_handled['SalePrice']

    module = importlib.import_module(config.bagging.base_model.module)
    base_model = getattr(module, config.bagging.base_model.name)(**config.bagging.base_model.params)

    bagging_model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', BaggingRegressor(estimator=base_model,
                                   random_state=config.general.seed,
                                   **config.bagging.params))
    ])

    scores = cross_val_score(bagging_model, X, y, cv=config.training.n_splits)
    model_acc = round(scores.mean(), 2)
    model_std = round(scores.std(), 2)

    bagging_model.fit(X, y)

    return bagging_model, model_acc, model_std


def test_bagging(test_data, model, model_name='bagging'):
    X_test = handling_for_linear(test_data)

    preds = model.predict(X_test)

    df = pd.DataFrame({'Id': test_data['Id'],
                       'SalePrice': preds})
    df.to_csv(path_or_buf=f'{config.paths.path_save_csv}{model_name}_preds.csv',
              index=False)


def train_stacking(train_data):
    train_data_handled = handling_for_linear(train_data)

    X = train_data_handled.drop(columns='SalePrice')
    y = train_data_handled['SalePrice']

    base_models = []
    for model in config.stacking.base_models:
        module = importlib.import_module(model.module)
        base_model = Pipeline([
            ('scale', preprocessor),
            ('model', getattr(module, model.name)(**model.params))
        ])
        base_models.append((model.name, base_model))

    module = importlib.import_module(config.stacking.meta_model.module)
    meta_model = Pipeline([
        ('scale', StandardScaler()),
        ('model', getattr(module, config.stacking.meta_model.name)(**config.stacking.meta_model.params))
    ])

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )

    scores = cross_val_score(stacking_model, X, y, cv=5)

    model_acc = round(scores.mean(), 2)
    model_std = round(scores.std(), 2)

    stacking_model.fit(X, y)
    return stacking_model, model_acc, model_std


def train_stacking_l2(train_data):
    train_data_handled = handling_for_linear(train_data)

    X = train_data_handled.drop(columns='SalePrice')
    y = train_data_handled['SalePrice']

    base_models = []
    for model in config.stacking_l2.base_models:
        module = importlib.import_module(model.module)
        base_model = Pipeline([
            ('scale', preprocessor),
            ('model', getattr(module, model.name)(**model.params))
        ])
        base_models.append((model.name, base_model))

    module = importlib.import_module(config.stacking_l2.meta_model.module)
    meta_model = Pipeline([
        ('scale', StandardScaler()),
        ('model', getattr(module, config.stacking_l2.meta_model.name)(**config.stacking_l2.meta_model.params))
    ])

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )

    scores = cross_val_score(stacking_model, X, y, cv=5)

    model_acc = round(scores.mean(), 2)
    model_std = round(scores.std(), 2)

    stacking_model.fit(X, y)
    return stacking_model, model_acc, model_std


def test_stacking(test_data, model, model_name='stacking'):
    X_test = handling_for_linear(test_data)

    preds = model.predict(X_test)

    df = pd.DataFrame({'Id': test_data['Id'],
                       'SalePrice': preds})
    df.to_csv(path_or_buf=f'{config.paths.path_save_csv}{model_name}_preds.csv',
              index=False)