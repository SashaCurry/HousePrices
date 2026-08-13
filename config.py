from omegaconf import OmegaConf
import numpy as np

config = {
    'general': {
        'experiment_name': 'HousePrices_v1.0',
        'seed': 0xFACED,
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
        'logreg': 0,
        'logreg_l1': 0,
        'logreg_l2': 0,
        'logreg_en': 0,
        'knn': 0,
        'dt': 0,
        'rf': 0,
        'catboost': 0,
        'lightgbm': 0,
        'xgboost': 0,
        'nn': 0,
        'bagging': 0,
        'stacking': 0,
        'stacking_l2': 0
    },
    'linreg': {
        'train_mode': True,
        'params': {
            'max_iter': 1000,
            'C': np.inf
        }
    },
    'linreg_l1': {
        'train_mode': True,
        'params': {
            'max_iter': 1000,
            'l1_ratio': 1.0,
            'solver': 'liblinear',
            'C': 1
        }
    },
    'linreg_l2': {
        'train_mode': True,
        'params': {
            'max_iter': 1000,
            'l1_ratio': 0.0,
            'solver': 'lbfgs',
            'C': 0.01
        }
    },
    'linreg_elnet': {
        'train_mode': True,
        'params': {
            'max_iter': 1000,
            'l1_ratio': 0.5,
            'solver': 'saga',
            'C': 1
        }
    },
    'knn': {
        'train_mode': True,
        'params': {
            'n_neighbors': 6,
            'weights': 'uniform',
            'metric': 'chebyshev'
        }
    },
    'decision_tree': {
        'train_mode': True,
        'params': {
            'max_depth': 4,
            'criterion': 'gini',
            'splitter': 'best'
        }
    },
    'random_forest': {
        'train_mode': True,
        'params': {
            'n_estimators': 50,
            'min_samples_leaf': 2
        }
    },
    'catboost': {
        'train_mode': True,
        'params': {
            'iterations': 100,
            'learning_rate': 0.075,
            'depth': 5,
            'loss_function': 'Logloss'
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
            'num_boost_round': 1000,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.75
        }
    },
    'neural_network': {
        'train_mode': True,
        'num_epochs': 20,
        'loss_fn': {
            'name': 'BCELoss',
            'params': {
            }
        },
        'optimizer': {
            'name': 'Adam',
            'params': {
                'lr': 0.001
            }
        },
        'scheduler': {
            'name': 'CosineAnnealingLR',
            'params': {
                'T_max': 100,
                'eta_min': 1e-6
            }
        }
    },
    'bagging': {
        'train_mode': True,
        'base_model': {
            'module': 'tree',
            'name': 'DecisionTreeClassifier',
            'params': {
                'max_depth': 6
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
                'module': 'neighbors',
                'name': 'KNeighborsClassifier',
                'params': {
                    'n_neighbors': 6
                }
            },
            {
                'module': 'tree',
                'name': 'DecisionTreeClassifier',
                'params': {
                    'max_depth': 3
                }
            },
            {
                'module': 'ensemble',
                'name': 'RandomForestClassifier',
                'params': {
                    'n_estimators': 100
                }
            }
        ],
        'meta_model': {
            'module': 'linear_model',
            'name': 'LogisticRegression',
            'params': {
                'max_iter': 1000
            }
        }
    },
    'stacking_l2': {
        'train_mode': True,
        'base_models': [
            {
                'module': 'neighbors',
                'name': 'KNeighborsClassifier',
                'params': {
                    'n_neighbors': 6
                }
            },
            {
                'module': 'tree',
                'name': 'DecisionTreeClassifier',
                'params': {
                    'max_depth': 3
                }
            },
            {
                'module': 'ensemble',
                'name': 'RandomForestClassifier',
                'params': {
                    'n_estimators': 100
                }
            }
        ],
        'meta_model': {
            'module': 'linear_model',
            'name': 'LogisticRegression',
            'params': {
                'max_iter': 1000,
                'l1_ratio': 0.0
            }
        }
    }
}

config = OmegaConf.create(config)
