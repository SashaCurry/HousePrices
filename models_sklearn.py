import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from config import config
from data_handle import *


def train_model_sklearn(X, y, model_name='logistic_regression'):
    X = preprocessing(X, handle_categorical='One-hot-encoding')

    model = None
    if model_name == 'logistic_regression':
        model = Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(**config.logreg.params))
        ])
    elif model_name == 'logistic_regression_l1':
        model = Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(**config.logreg_l1.params))
        ])
    elif model_name == 'logistic_regression_l2':
        model = Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(**config.logreg_l2.params))
        ])
    elif model_name == 'logistic_regression_elasticnet':
        model = Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(**config.logreg_elnet.params))
        ])
    elif model_name == 'knn':
        model = Pipeline([
            ('scale', StandardScaler()),
            ('model', KNeighborsClassifier(**config.knn.params))
        ])
    elif model_name == 'decision_tree':
        model = DecisionTreeClassifier(**config.decision_tree.params)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(**config.random_forest.params,
                                       random_state=config.general.seed)

    skf = StratifiedKFold(n_splits=config.training.n_splits, shuffle=True, random_state=config.general.seed)
    scores = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)

        cur_accuracy = model.score(X_val, y_val)
        scores.append(cur_accuracy)

    model_acc = round(sum(scores) / len(scores), 2)
    model_std = round(np.array(scores).std(), 2)

    model.fit(X, y)
    return model, model_acc, model_std


def test_model_sklearn(X, model, model_name):
    X_test = preprocessing(X, handle_categorical='One-hot-encoding')

    preds = model.predict(X_test)

    df = pd.DataFrame({'PassengerId': X['PassengerId'],
                       'Survived': preds})
    df.to_csv(path_or_buf=f'{config.paths.path_save_csv}{model_name}_preds.csv',
              index=False)