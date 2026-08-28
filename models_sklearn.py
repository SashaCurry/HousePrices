from sklearn.model_selection import KFold
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

    model = None
    if model_name == 'linear_regression':
        model = Pipeline([
            ('scale', preprocessor),
            ('model', SGDRegressor(**config.linreg.params))
        ])
    elif model_name == 'linear_regression_l1':
        model = Pipeline([
            ('scale', preprocessor),
            ('model', SGDRegressor(**config.linreg_l1.params))
        ])
    elif model_name == 'linear_regression_l2':
        model = Pipeline([
            ('scale', preprocessor),
            ('model', SGDRegressor(**config.linreg_l2.params))
        ])
    elif model_name == 'linear_regression_elasticnet':
        model = Pipeline([
            ('scale', preprocessor),
            ('model', SGDRegressor(**config.linreg_elnet.params))
        ])
    elif model_name == 'knn':
        model = Pipeline([
            ('scale', preprocessor),
            ('model', KNeighborsRegressor(**config.knn.params))
        ])
    elif model_name == 'decision_tree':
        model = Pipeline([
            ('scale', preprocessor),
            ('model', DecisionTreeRegressor(**config.decision_tree.params))
        ])
    elif model_name == 'random_forest':
        model = Pipeline([
            ('scale', preprocessor),
            ('model', RandomForestRegressor(**config.random_forest.params))
        ])

    kf = KFold(n_splits=config.training.n_splits, shuffle=True)
    scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)

        cur_accuracy = model.score(X_val, y_val)
        scores.append(cur_accuracy)

    model_acc = round(sum(scores) / len(scores), 2)
    model_std = round(np.array(scores).std(), 2)

    model.fit(X, y)
    return model, model_acc, model_std


def test_model_sklearn(data, model, model_name):
    test_data_handled = handling_for_linear(data)
    X_test = test_data_handled.drop(columns=['SalePrice'])

    preds = model.predict(X_test)

    df = pd.DataFrame({'PassengerId': X_test['Id'],
                       'SalePrice': preds})
    df.to_csv(path_or_buf=f'{config.paths.path_save_csv}{model_name}_preds.csv',
              index=False)