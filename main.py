import os
import random
import joblib

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import torch

from config import config
from models_sklearn import *


def run(config):
    train_data = pd.read_csv(config.paths.path_to_train)
    test_data = pd.read_csv(config.paths.path_to_test)

    X = train_data.drop(columns=['SalePrice'])
    y = train_data['SalePrice']

    # Суммарная информация о всех моделях и их метриках
    model_data = []

    # ↓↓↓ Логистическая регрессия ↓↓↓

    linreg_model, linreg_acc, linreg_std = None, '—', '—'
    if config.logreg.train_mode:
        linreg_model, linreg_acc, linreg_std = train_model_sklearn(X, y, model_name='linear_regression')
        joblib.dump(linreg_model, config.paths.path_save_models + 'linreg_model.joblib')
    else:
        try:
            linreg_model = joblib.load(config.paths.path_save_models + 'linreg_model.joblib')
        except FileNotFoundError:
            print(f'Модель linreg не загружена. '
                  f'Проверьте наличие файла "linreg_model.joblib" в {config.paths.path_save_models}')

    model_data.append(['LinReg', linreg_acc, linreg_std, config.lb_scores.linreg])
    # test_model_sklearn(X=test_data, model=linreg_model, model_name='linreg')
    '''
    # ↓↓↓ Логистическая регрессия с L1-регуляризацией ↓↓↓

    logreg_l1_model, logreg_l1_acc, logreg_l1_std = None, '—', '—'
    if config.logreg_l1.train_mode:
        logreg_l1_model, logreg_l1_acc, logreg_l1_std = train_model_sklearn(X, y, model_name='logistic_regression_l1')
        joblib.dump(logreg_l1_model, config.paths.path_save_models + 'logreg_l1_model.joblib')
    else:
        try:
            logreg_l1_model = joblib.load(config.paths.path_save_models + 'logreg_l1_model.joblib')
        except FileNotFoundError:
            print(f'Модель logreg_l1 не загружена. '
                  f'Проверьте наличие файла "logreg_l1_model.joblib" в {config.paths.path_save_models}')

    model_data.append(['LogReg-L1', logreg_l1_acc, logreg_l1_std, config.lb_scores.logreg_l1])
    test_model_sklearn(X=test_data, model=logreg_l1_model, model_name='logreg_l1')

    # ↓↓↓ Логистическая регрессия с L2-регуляризацией ↓↓↓

    logreg_l2_model, logreg_l2_acc, logreg_l2_std = None, '—', '—'
    if config.logreg_l2.train_mode:
        logreg_l2_model, logreg_l2_acc, logreg_l2_std = train_model_sklearn(X, y, model_name='logistic_regression_l2')
        joblib.dump(logreg_l2_model, config.paths.path_save_models + 'logreg_l2_model.joblib')
    else:
        try:
            logreg_l2_model = joblib.load(config.paths.path_save_models + 'logreg_l2_model.joblib')
        except FileNotFoundError:
            print(f'Модель logreg_l2 не загружена. '
                  f'Проверьте наличие файла "logreg_l2_model.joblib" в {config.paths.path_save_models}')

    model_data.append(['LogReg-L2', logreg_l2_acc, logreg_l2_std, config.lb_scores.logreg_l2])
    test_model_sklearn(X=test_data, model=logreg_l2_model, model_name='logreg_l2')

    # ↓↓↓ Логистическая регрессия с ElasticNet-регуляризацией ↓↓↓

    logreg_en_model, logreg_en_acc, logreg_en_std = None, '—', '—'
    if config.logreg_elnet.train_mode:
        logreg_en_model, logreg_en_acc, logreg_en_std = train_model_sklearn(X, y,
                                                                            model_name='logistic_regression_elasticnet')
        joblib.dump(logreg_en_model, config.paths.path_save_models + 'logreg_en_model.joblib')
    else:
        try:
            logreg_en_model = joblib.load(config.paths.path_save_models + 'logreg_en_model.joblib')
        except FileNotFoundError:
            print(f'Модель logreg_en не загружена. '
                  f'Проверьте наличие файла "logreg_en_model.joblib" в {config.paths.path_save_models}')

    model_data.append(['LogReg-ElNet', logreg_en_acc, logreg_en_std, config.lb_scores.logreg_en])
    test_model_sklearn(X=test_data, model=logreg_en_model, model_name='logreg_en')

    # ↓↓↓ Метод ближайших соседей KNN ↓↓↓

    knn_model, knn_acc, knn_std = None, '—', '—'
    if config.knn.train_mode:
        knn_model, knn_acc, knn_std = train_model_sklearn(X, y, model_name='knn')
        joblib.dump(knn_model, config.paths.path_save_models + 'knn_model.joblib')
    else:
        try:
            knn_model = joblib.load(config.paths.path_save_models + 'knn_model.joblib')
        except FileNotFoundError:
            print(f'Модель knn не загружена. '
                  f'Проверьте наличие файла "knn_model.joblib" в {config.paths.path_save_models}')

    model_data.append(['KNN', knn_acc, knn_std, config.lb_scores.knn])
    test_model_sklearn(X=test_data, model=knn_model, model_name='knn')

    # ↓↓↓ Решающее дерево DecisionTree ↓↓↓

    dt_model, dt_acc, dt_std = None, '—', '—'
    if config.decision_tree.train_mode:
        dt_model, dt_acc, dt_std = train_model_sklearn(X, y, model_name='decision_tree')
        joblib.dump(dt_model, config.paths.path_save_models + 'dt_model.joblib')
    else:
        try:
            dt_model = joblib.load(config.paths.path_save_models + 'dt_model.joblib')
        except FileNotFoundError:
            print(f'Модель decision tree не загружена. '
                  f'Проверьте наличие файла "dt_model.joblib" в {config.paths.path_save_models}')

    model_data.append(['DecisionTree', dt_acc, dt_std, config.lb_scores.dt])
    test_model_sklearn(X=test_data, model=dt_model, model_name='decision_tree')

    # ↓↓↓ Случайный лес RandomForest ↓↓↓

    rf_model, rf_acc, rf_std = None, '—', '—'
    if config.random_forest.train_mode:
        rf_model, rf_acc, rf_std = train_model_sklearn(X, y, model_name='random_forest')
        joblib.dump(rf_model, config.paths.path_save_models + 'rf_model.joblib')
    else:
        try:
            rf_model = joblib.load(config.paths.path_save_models + 'rf_model.joblib')
        except FileNotFoundError:
            print(f'Модель random forest не загружена. '
                  f'Проверьте наличие файла "rf_model.joblib" в {config.paths.path_save_models}')

    model_data.append(['RandomForest', rf_acc, rf_std, config.lb_scores.rf])
    test_model_sklearn(X=test_data, model=rf_model, model_name='random_forest')

    # ↓↓↓ Бустинг CatBoost ↓↓↓

    catboost_model, catboost_acc, catboost_std = None, '—', '—'
    if config.catboost.train_mode:
        catboost_model, catboost_acc, catboost_std = train_catboost(X, y)
        catboost_model.save_model(config.paths.path_save_models + 'catboost_model.cbm')
    else:
        try:
            catboost_model = CatBoostClassifier()
            catboost_model = catboost_model.load_model(config.paths.path_save_models + 'catboost_model.cbm')
        except FileNotFoundError:
            print(f'Модель catboost не загружена. '
                  f'Проверьте наличие файла "catboost_model.cbm" в {config.paths.path_save_models}')

    model_data.append(['CatBoost', catboost_acc, catboost_std, config.lb_scores.catboost])
    test_boost(X=test_data, model=catboost_model, model_name='catboost')

    # ↓↓↓ Бустинг LightGBM ↓↓↓

    lightgbm_model, lightgbm_acc, lightgbm_std = None, '—', '—'
    if config.lightgbm.train_mode:
        lightgbm_model, lightgbm_acc, lightgbm_std = train_lightgbm(X, y)
        lightgbm_model.booster_.save_model(config.paths.path_save_models + 'lightgbm_model.txt')
    else:
        try:
            lightgbm_model = lgb.Booster(model_file=config.paths.path_save_models + 'lightgbm_model.txt')
        except FileNotFoundError:
            print(f'Модель lightgbm не загружена. '
                  f'Проверьте наличие файла "lightgbm_model.txt" в {config.paths.path_save_models}')

    model_data.append(['LightGBM', lightgbm_acc, lightgbm_std, config.lb_scores.lightgbm])
    test_boost(X=test_data, model=lightgbm_model, model_name='lightgbm')

    # ↓↓↓ Бустинг XGBoost ↓↓↓

    xgboost_model, xgboost_acc, xgboost_std = None, '—', '—'
    if config.xgboost.train_mode:
        xgboost_model, xgboost_acc, xgboost_std = train_xgboost(X, y)
        xgboost_model.save_model(config.paths.path_save_models + 'xgboost_model.json')
    else:
        try:
            xgboost_model = xgb.XGBClassifier()
            xgboost_model.load_model(config.paths.path_save_models + 'xgboost_model.json')
        except FileNotFoundError:
            print(f'Модель xgboost не загружена. '
                  f'Проверьте наличие файла "xgboost_model.json" в {config.paths.path_save_models}')

    model_data.append(['XGBoost', xgboost_acc, xgboost_std, config.lb_scores.xgboost])
    test_boost(X=test_data, model=xgboost_model, model_name='xgboost')

    # ↓↓↓ Нейронная сеть ↓↓↓

    nn_model, nn_acc = None, "—"
    if config.neural_network.train_mode:
        nn_model, nn_acc = train_nn(X, y)
        nn_model.to('cpu')
        torch.save(nn_model, config.paths.path_save_models + 'nn_model.pt')
    else:
        try:
            nn_model = torch.load(f=config.paths.path_save_models + 'nn_model.pt',
                                  weights_only=False)
        except FileNotFoundError:
            print(f'Модель neural network не загружена. '
                  f'Проверьте наличие файла "nn_model.joblib" в {config.paths.path_save_models}')

    model_data.append(['NeuralNetwork', nn_acc, '—', config.lb_scores.nn])
    test_nn(X=test_data, model=nn_model)

    # ↓↓↓ Ансамбль Bagging ↓↓↓

    bagging_model, bagging_acc, bagging_std = None, '—', '—'
    if config.bagging.train_mode:
        bagging_model, bagging_acc, bagging_std = train_bagging(X, y)
        joblib.dump(bagging_model, config.paths.path_save_models + 'bagging_model.joblib')
    else:
        try:
            bagging_model = joblib.load(config.paths.path_save_models + 'bagging_model.joblib')
        except FileNotFoundError:
            print(f'Модель bagging не загружена. '
                  f'Проверьте наличие файла "bagging_model.joblib" в {config.paths.path_save_models}')

    model_data.append(['Bagging', bagging_acc, bagging_std, config.lb_scores.bagging])
    test_bagging(X=test_data, model=bagging_model)

    # ↓↓↓ Ансамбль Stacking via LogReg ↓↓↓

    stacking_model, stacking_acc, stacking_std = None, '—', '—'
    if config.stacking.train_mode:
        stacking_model, stacking_acc, stacking_std = train_stacking(X, y)
        joblib.dump(stacking_model, config.paths.path_save_models + 'stacking_model.joblib')
    else:
        try:
            stacking_model = joblib.load(config.paths.path_save_models + 'stacking_model.joblib')
        except FileNotFoundError:
            print(f'Модель stacking не загружена. '
                  f'Проверьте наличие файла "stacking_model.joblib" в {config.paths.path_save_models}')

    model_data.append(['Stacking via LogReg', stacking_acc, stacking_std, config.lb_scores.stacking])
    test_stacking(X=test_data, models=stacking_model, model_name='stacking')

    # ↓↓↓ Ансамбль Stacking via LogReg-L2 ↓↓↓

    stacking_l2_model, stacking_l2_acc, stacking_l2_std = None, '—', '—'
    if config.stacking_l2.train_mode:
        stacking_l2_model, stacking_l2_acc, stacking_l2_std = train_stacking_l2(X, y)
        joblib.dump(stacking_l2_model, config.paths.path_save_models + 'stacking_l2_model.joblib')
    else:
        try:
            stacking_l2_model = joblib.load(config.paths.path_save_models + 'stacking_l2_model.joblib')
        except FileNotFoundError:
            print(f'Модель stacking_l2 не загружена. '
                  f'Проверьте наличие файла "stacking_l2_model.joblib" в {config.paths.path_save_models}')

    model_data.append(['Stacking via LogReg-L2', stacking_l2_acc, stacking_l2_std, config.lb_scores.stacking_l2])
    test_stacking(X=test_data, models=stacking_l2_model, model_name='stacking_l2')
    '''
    # Total output

    header = f'{"Approach":<22} | {"CV":>10} | {"CV STD":>10} | {"LB":>10}'
    print(header)
    print('-' * len(header))

    for item in model_data:
        DEFAULT_VAUES = ['—', '—', '—']
        model_name, model_acc, model_std, model_lb = (item + DEFAULT_VAUES)[:4]
        print(f'{model_name:<22} | {model_acc:>10} | {model_std:>10} | {model_lb:>10}')


def main(config):
    # Устанавливаем детерменированность
    random.seed(config.general.seed)
    np.random.seed(config.general.seed)
    torch.manual_seed(config.general.seed)
    torch.cuda.manual_seed(config.general.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(config.general.seed)

    # Создаём директории для работы с программой
    os.makedirs(f'{config.paths.path_save_csv}', exist_ok=True)
    os.makedirs(f'{config.paths.path_save_models}', exist_ok=True)

    run(config)


if __name__ == "__main__":
    main(config)