import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from config import config
from data_handle import *

#TODO: Вопрос из Титаника остался открытым, я забил на него
class HousePricesNN(nn.Module):
    def __init__(self, input):
        super().__init__()

        self.layer_1 = nn.Linear(input, 100)
        self.layer_2 = nn.Linear(100, 1)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.activation(x)
        out = self.layer_2(x)

        return out


def train_nn(train_data):
    train_data_handled = handling_for_linear(train_data)

    X_raw = train_data_handled.drop(columns='SalePrice')
    y_raw = train_data_handled['SalePrice'].values

    # Разделение до тензоров, для защиты от утечки данных
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_raw, y_raw, test_size=0.2)

    target_pipeline = Pipeline([
        ('encoder', TargetEncoder(target_type='continuous')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('target', target_pipeline, ['BsmtFinType1', 'TotalBaths', 'FireplaceQu', 'GarageType', 'SaleCondition']),
            ('cat', MinMaxScaler(), ['MSSubClass_Rating', 'LotConfig_Rating', 'Neighborhood_Rating',
                                     'Condition1_Rating', 'OverallQual', 'Exterior1st_Rating', 'ExterQual_Rating',
                                     'BsmtQual_Rating', 'BsmtExposure_Rating', 'BsmtFinSF_Ratio', 'KitchenQual_Rating',
                                     'TotRmsAbvGrd_Rating', 'Fireplaces_Rating', 'GarageCars_Rating']),
            ('num', StandardScaler(), ['LotFrontage', 'LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageAge',
                                       'GarageArea', 'WoodDeckSF', 'HouseAge', 'RemodAge'])
        ],
        remainder='passthrough'
    )

    y_scaler = MinMaxScaler()

    # Обучаем препроцессор на тренировочных данных, а валидационные просто трансформируем
    X_train_scaled = preprocessor.fit_transform(X_train_raw, y_train)
    X_val_scaled = preprocessor.transform(X_val_raw)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))

    # Превращаем в тензоры
    X_train_tensor = torch.tensor(data=X_train_scaled,
                                  dtype=torch.float32,
                                  device=config.training.device)
    y_train_tensor = torch.tensor(data=y_train_scaled,
                                  dtype=torch.float32,
                                  device=config.training.device).view(-1, 1)

    X_val_tensor = torch.tensor(data=X_val_scaled,
                                dtype=torch.float32,
                                device=config.training.device)
    y_val_tensor = torch.tensor(data=y_val_scaled,
                                dtype=torch.float32,
                                device=config.training.device).view(-1, 1)

    # Создание датасетов и даталоадеров
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    val_data = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # Создание параметров для обучения
    model = HousePricesNN(input=X_train_tensor.shape[1]).to(config.training.device)
    loss_fn = getattr(nn, config.neural_network.loss_fn.name)()
    optimizer = getattr(torch.optim, config.neural_network.optimizer.name)(model.parameters(), **config.neural_network.optimizer.params)
    scheduler = getattr(torch.optim.lr_scheduler, config.neural_network.scheduler.name)(optimizer, **config.neural_network.scheduler.params)

    num_epochs = config.neural_network.num_epochs

    # ЦИКЛ ОБУЧЕНИЯ
    mean_val_acc = 0
    for epoch in range(num_epochs):
        # ТРЕНИРОВКА
        model.train()
        for X_batch, y_batch in train_loader:
            # Прямой проход + расчёт ошибки модел
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)

            # Обратный проход
            optimizer.zero_grad()
            loss.backward()

            # Шаг оптимизатора
            optimizer.step()

        # ВАЛИДАЦИЯ
        all_preds = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Прямой проход
                pred = model(X_batch)

                # Сохраняем предсказания, чтобы потом рассчитать R^2
                all_preds.extend(pred.cpu().numpy())

        # R^2 на валидационной выборке
        mean_val_acc = r2_score(y_val_tensor.cpu().numpy(), all_preds)

        scheduler.step()

    return model, round(mean_val_acc, 2)