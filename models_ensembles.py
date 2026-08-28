import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
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


def train_stacking(train_data):
    train_data_handled_for_linear = handling_for_linear(train_data)

