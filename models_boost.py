import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from config import config
from data_handle import *

CAT_FEATURES = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'MoSold', 'YrSold',
                'SaleType', 'SaleCondition']


def train_catboost(train_data):
    train_data_handled = handling_for_boosting(train_data)

    X = train_data_handled.drop(columns=['SalePrice'])
    y = train_data_handled[['SalePrice']]

    model = cb.CatBoostRegressor(**config.catboost.params,
                                 verbose=False)

    cv_scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        cv=config.training.n_splits,
        scoring='neg_root_mean_squared_log_error',
        params={'cat_features': CAT_FEATURES}
    )

    model_acc = round(-cv_scores.mean(), 2)
    model_std = round(cv_scores.std(), 2)

    model = cb.CatBoostRegressor(
        **config.catboost.params,
        cat_features=CAT_FEATURES
    )
    model.fit(X, y, verbose=False)

    return model, model_acc, model_std


def train_lightgbm(train_data):
    train_data_handled = handling_for_boosting(train_data)
    train_data_handled[CAT_FEATURES] = train_data_handled[CAT_FEATURES].astype('category')

    X = train_data_handled.drop(columns=['SalePrice'])
    y = train_data_handled[['SalePrice']]

    model = LGBMRegressor(
        objective='regression',
        **config.lightgbm.params,
        verbose=-1
    )

    cv_scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        cv=config.training.n_splits,
        scoring='neg_root_mean_squared_log_error'
    )

    model_acc = round(-cv_scores.mean(), 2)
    model_std = round(cv_scores.std(), 2)

    model.fit(X, y)

    return model, model_acc, model_std


def train_xgboost(train_data):
    train_data_handled = handling_for_boosting(train_data)
    train_data_handled[CAT_FEATURES] = train_data_handled[CAT_FEATURES].astype('category')

    X = train_data_handled.drop(columns=['SalePrice'])
    y = train_data_handled[['SalePrice']]

    model = XGBRegressor(
        **config.xgboost.params,
        tree_method='hist',
        enable_categorical=True
    )

    cv_scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        cv=config.training.n_splits,
        scoring='neg_root_mean_squared_log_error'
    )

    model_acc = round(-cv_scores.mean(), 2)
    model_std = round(cv_scores.std(), 2)

    model.fit(X, y)

    return model, model_acc, model_std


def test_boost(data, model, model_name):
    X_test = handling_for_boosting(data)
    X_test[CAT_FEATURES] = X_test[CAT_FEATURES].astype('category')

    preds = model.predict(X_test)

    df = pd.DataFrame({'Id': data['Id'],
                       'SalePrice': preds})
    df.to_csv(path_or_buf=f'{config.paths.path_save_csv}{model_name}_preds.csv',
              index=False)