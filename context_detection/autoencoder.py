import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# 定义多层Autoencoder模型
class MultiLayerAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, encoding_size):
        super(MultiLayerAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(True),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(True),
            nn.Linear(hidden_size2, encoding_size),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, hidden_size2),
            nn.ReLU(True),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(True),
            nn.Linear(hidden_size1, input_size),
            # nn.Sigmoid()  # 使用Sigmoid激活函数确保输出在[0,1]之间
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 假设的数据维度


background_power_demands = np.load("../simulator/data/excluded_power_no_renew_kWh_14_15.npy")
agg_background_power_demands = np.load("../simulator/data/new_agg_no_renew_kWh_14_15.npy")
renewable_generations = np.load("../simulator/data/renewable_generation_kW_14_15_London.npy")
# print(renewable_generations)
start_hour = 0
day_span = 7
day_data = []
weekly_win_data = []
for i in range(start_hour, len(renewable_generations), 24):
    day_data.append(renewable_generations[i:i + 24])
# print(f"len:{len(day_data)}")
day = 0
while day < 364:
    weekly_win_data.append(day_data[day:day + 7])
    day += 7
    # print(day_data[day:day + 7])
# print(f"len:{len(weekly_win_data)}")
weekly_win_data = np.array(weekly_win_data)


def train_autoencoder(weekly_win_data, start_week, end_week):
    input_size = 24  # 输入特征的数量
    hidden_size1 = 64
    hidden_size2 = 32
    encoding_size = 16

    # 实例化模型
    model = MultiLayerAutoencoder(input_size, hidden_size1, hidden_size2, encoding_size)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data = torch.from_numpy(weekly_win_data[start_week:end_week])
    labels = torch.from_numpy(weekly_win_data[start_week:end_week])
    # 创建数据加载器
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # 训练过程
    epochs = 500
    for epoch in range(epochs):
        for data, labels in dataloader:
            data = data.float()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
        # print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
    print("finish training")
    return model


def context_change_detect(model, starting_week, weekly_win_data):
    loss_benches = []
    criterion = nn.MSELoss()

    for i in range(starting_week, starting_week + 4):
        test_data = torch.from_numpy(weekly_win_data[i])
        test_data = test_data.float()
        outputs = model(test_data)
        loss = criterion(outputs, test_data)
        loss_benches.append(loss)
    loss_bench = max(loss_benches)

    for i in range(starting_week, 52):
        test_data = torch.from_numpy(weekly_win_data[i])
        test_data = test_data.float()
        outputs = model(test_data)
        loss = criterion(outputs, test_data)
        print(f"week:{i + 1},month:{(i // 4) + 1}\tloss:{loss},loss_bench:{loss_bench}")
        if loss > loss_bench:
            return i


anomaly_week = 0
while True:
    if anomaly_week == 0:
        model = train_autoencoder(weekly_win_data=weekly_win_data, start_week=0, end_week=4)
        anomaly_week = context_change_detect(model, 0, weekly_win_data)
        print(f"anomaly_week:{anomaly_week}")
    else:
        if anomaly_week + 2 > 52:
            break
        model = train_autoencoder(weekly_win_data=weekly_win_data, start_week=anomaly_week - 2,
                                  end_week=anomaly_week + 2)
        anomaly_week = context_change_detect(model, anomaly_week - 2, weekly_win_data)
        print(f"anomaly_week:{anomaly_week}")
    if anomaly_week is None:
        break
