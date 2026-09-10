from omegaconf import OmegaConf

config = {
    'general': {
        'experiment_name': 'HousePrices_v1.0',
        'seed': 0x555
    },
    'paths': {
        'path_to_train': './content/data/train.csv',
        'path_to_test': './content/data/test.csv',
        'path_save_csv': './content/preds/',
        'path_save_models': './content/models/'
    },
    'training': {
        'device': 'cpu',
        'n_splits': 5
    },
    'lb_scores': {
        'linreg': 0.21,
        'linreg_l1': 0.22,
        'linreg_l2': 0.22,
        'linreg_en': 0.18,
        'knn': 0.17,
        'dt': 0.23,
        'rf': 0.15,
        'catboost': 0.14,
        'lightgbm': 0.14,
        'xgboost': 0.13,
        'nn': 0.15,
        'bagging': 0.15,
        'stacking': 0.13,
        'stacking_l2': 0.13
    },
    'linreg': {
        'train_mode': True,
        'params': {
            'loss': 'squared_error',
            'learning_rate': 'adaptive',
            'eta0': 0.01,
            'max_iter': 1000,
            'penalty': None,
        }
    },
    'linreg_l1': {
        'train_mode': True,
        'params': {
            'loss': 'squared_error',
            'learning_rate': 'adaptive',
            'eta0': 0.01,
            'max_iter': 1000,
            'penalty': 'l1',
            'alpha': 0.0001,
        }
    },
    'linreg_l2': {
        'train_mode': True,
        'params': {
            'loss': 'squared_error',
            'learning_rate': 'adaptive',
            'eta0': 0.01,
            'max_iter': 1000,
            'alpha': 0.0001
        }
    },
    'linreg_elnet': {
        'train_mode': True,
        'params': {
            'loss': 'squared_error',
            'learning_rate': 'adaptive',
            'eta0': 0.01,
            'max_iter': 1000,
            'l1_ratio': 0.5,
            'alpha': 0.0001
        }
    },
    'knn': {
        'train_mode': True,
        'params': {
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'euclidean'
        }
    },
    'decision_tree': {
        'train_mode': True,
        'params': {
            'max_depth': 4,
            'criterion': 'squared_error',
            'splitter': 'best',
            'min_samples_split': 5,
        }
    },
    'random_forest': {
        'train_mode': True,
        'params': {
            'n_estimators': 50,
            'max_depth': 10,
        }
    },
    'catboost': {
        'train_mode': True,
        'params': {
            'iterations': 100,
            'learning_rate': 0.075,
            'depth': 5,
            'loss_function': 'RMSE'
        }
    },
    'lightgbm': {
        'train_mode': True,
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 31
        }
    },
    'xgboost': {
        'train_mode': True,
        'params': {
            'n_estimators': 1000,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.75
        }
    },
    'neural_network': {
        'train_mode': True,
        'num_epochs': 50,
        'batch_size': 32,
        'loss_fn': {
            'name': 'MSELoss',
            'params': {
            }
        },
        'optimizer': {
            'name': 'AdamW',
            'params': {
                'lr': 0.001,
            }
        },
        'scheduler': {
            'name': 'ReduceLROnPlateau',
            'params': {
                'mode': 'min',
                'factor': 0.5,
                'patience': 5
            }
        }
    },
    'bagging': {
        'train_mode': True,
        'base_model': {
            'module': 'sklearn.tree',
            'name': 'DecisionTreeRegressor',
            'params': {
                'max_depth': 15
            }
        },
        'params': {
            'n_estimators': 100
        }
    },
    'stacking': {
        'train_mode': True,
        'base_models': [
            {
                'module': 'sklearn.linear_model',
                'name': 'SGDRegressor',
                'params': {
                    'loss': 'squared_error',
                    'penalty': 'l2',
                    'alpha': 0.001,
                    'max_iter': 1000
                }
            },
            {
                'module': 'sklearn.ensemble',
                'name': 'RandomForestRegressor',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 3
                }
            },
            {
                'module': 'sklearn.neighbors',
                'name': 'KNeighborsRegressor',
                'params': {
                    'n_neighbors': 6
                }
            },
            {
                'module': 'catboost',
                'name': 'CatBoostRegressor',
                'params': {
                    'iterations': 100,
                    'depth': 5,
                    'loss_function': 'RMSE',
                    'verbose': 0
                }
            }
        ],
        'meta_model': {
            'module': 'sklearn.linear_model',
            'name': 'SGDRegressor',
            'params': {
                'max_iter': 1000,
            }
        }
    },
    'stacking_l2': {
        'train_mode': True,
        'base_models': [
            {
                'module': 'sklearn.linear_model',
                'name': 'SGDRegressor',
                'params': {
                    'loss': 'squared_error',
                    'penalty': 'l2',
                    'alpha': 0.001,
                    'max_iter': 1000
                }
            },
            {
                'module': 'sklearn.ensemble',
                'name': 'RandomForestRegressor',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 3
                }
            },
            {
                'module': 'sklearn.neighbors',
                'name': 'KNeighborsRegressor',
                'params': {
                    'n_neighbors': 6
                }
            },
            {
                'module': 'catboost',
                'name': 'CatBoostRegressor',
                'params': {
                    'iterations': 100,
                    'depth': 5,
                    'loss_function': 'RMSE',
                    'verbose': 0
                }
            }
        ],
        'meta_model': {
            'module': 'sklearn.linear_model',
            'name': 'SGDRegressor',
            'params': {
                'max_iter': 1000,
                'penalty': 'l2',
                'alpha': 0.01
            }
        }
    }
}

config = OmegaConf.create(config)