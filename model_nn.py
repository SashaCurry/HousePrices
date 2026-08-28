import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from config import config
from data_handle import *

#TODO: Вопрос из Титаника остался открытым, я забил на него
#      Я не то что забил, я решил вообще не реализовывать Embedding-слои
class HousePricesNN(nn.Module):
    def __init__(self, input):
        super().__init__()

        self.layer_1 = nn.Linear(input, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(32, 16)
        self.layer_5 = nn.Linear(16, 8)
        self.layer_6 = nn.Linear(8, 1)

        self.activation = nn.LeakyReLU()

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer_3(x)
        x = self.activation(x)
        x = self.layer_4(x)
        x = self.activation(x)
        x = self.layer_5(x)
        x = self.activation(x)
        out = self.layer_6(x)

        return out


def train_nn(train_data):
    train_data_handled = handling_for_linear(train_data)

    X_raw = train_data_handled.drop(columns='SalePrice')
    y_raw = train_data_handled['SalePrice'].values

    # Разделение перед target-encoding, для защиты от утечки данных
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_raw, y_raw,
                                                              test_size=0.25,
                                                              random_state=config.general.seed)

    # Таргет тоже надо нормализовывать
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

    train_loader = DataLoader(train_data, batch_size=config.neural_network.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.neural_network.batch_size, shuffle=False)

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
        val_loss = 0.0
        all_preds = []

        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Прямой проход
                pred = model(X_batch)
                loss = loss_fn(pred, y_batch)

                val_loss += loss.item()

                # Сохраняем предсказания, чтобы потом рассчитать R^2
                all_preds.extend(pred.cpu().numpy())

        # R^2 на валидационной выборке
        mean_val_acc = r2_score(y_val_tensor.cpu().numpy(), all_preds)

        # Шаг планировщика
        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)

    return model, round(mean_val_acc, 2)


## TODO: при реализации тестирование не забыть отмасштабировать таргет обратно