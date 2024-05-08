import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import copy
# %%

# 固定随机种子，设置device
seed = 3407

np.random.seed(seed)
torch.manual_seed(seed)

device = 'cuda:0'
# device = 'mps'  # for Apple
# torch.set_default_device(device)
torch.cuda.set_device(device)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
# %% md

### 2.1 模型定义

# %%

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# %%

class BiLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # Bi-LSTM 需要两个隐藏状态
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# %%

class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# %% md

### 2.2 DataLoader

# %%

def createSequence(df, window_size):
    # 找到轨迹长度 >= 15的所有轨迹
    # 按照滑动窗口进行划分


    seq = []
    label = []
    df = df.drop(columns = 'Unnamed: 0')
    df = df.loc[~(df == 0).all(axis=1)]
    df = df.rolling(window=5).mean().bfill()  #平滑处理
    trajectory = df.values.tolist()
    traj = copy.deepcopy(trajectory)
    x = []
    for tra in traj:
        sum = 0
        for tr in tra:
            sum+=tr
        if sum!=0:
            for i in range(len(tra)):
                if tra[i] == 0:
                    if i == 0:
                        tra[i] = tra[i+1]
                    elif i == len(tra) - 1:
                        tra[i] = tra[i-1]
                    else:
                        tra[i] = (tra[i-1]+tra[i+1])/2
            x.append(tra)
    trajectory = x

    num_splits = len(trajectory) - window_size + 1

    for i in range(num_splits):
        seq.append(trajectory[i:i + window_size - 1])
        label.append(trajectory[i + window_size - 1])

    seq = torch.tensor(np.array(seq), dtype=torch.float32).to(device)
    label = torch.tensor(np.array(label), dtype=torch.float32).to(device)
    return seq, label


# %% md

### 2.3 训练 & 评估定义

# %%

# 计算 RMSE（只计算点坐标和距离，不包括时间、速度等）
def calc_rmse(predictions, targets):
    mse = torch.mean((predictions[:, :3] - targets[:, :3]) ** 2)
    rmse = torch.sqrt(mse)
    return rmse


# %%

def trainModel(train_X, train_Y, val_X, val_Y, model,
               lr=1e-2, epoch_num=20, logging_steps=5):
    # 使用 DataLoader 和 TensorDataset 批量加载数据
    train_set = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, generator=torch.Generator(device=device))
    test_set = TensorDataset(val_X, val_Y)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, generator=torch.Generator(device=device))

    # 初始化损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 模型训练
    for epoch in tqdm(range(epoch_num)):
        loss = 0.0
        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss += loss.item()

        # log
        if epoch % logging_steps == (logging_steps - 1):
            print(f"Epoch {epoch + 1}, loss: {loss / len(train_loader):.6f}")

    # 用验证集评估
    preds = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)

            preds.append(outputs)

    preds = torch.cat(preds, dim=0)
    loss = criterion(preds, val_Y)
    rmse = calc_rmse(preds, val_Y)

    return model, loss.item(), rmse.item()


def split_dataset(df, window_size):
    seq, label = createSequence(df, window_size)

    # train
    train_seq, test_seq = train_test_split(seq, test_size=0.4, random_state=seed)
    train_label, test_label = train_test_split(label, test_size=0.4, random_state=seed)

    # val & test
    val_seq, test_seq = train_test_split(test_seq, test_size=0.5, random_state=seed)
    val_label, test_label = train_test_split(test_label, test_size=0.5, random_state=seed)

    return train_seq, train_label, val_seq, val_label, test_seq, test_label

df = pd.read_csv('G:\北京理工大学学习课程\学期二\数据挖掘\Beijing-Traffic-Track-Data-Mining-master\Beijing-Traffic-Track-Data-Mining-master\METR-LA.csv')

seq ,label = createSequence(df,15)



# %%

# 部分参数
window_size = 10
lr = 1e-4
epoch_num = 200
logging_steps = 10

# %%

# 尝试多种模型与特征
best_model = None
best_features = []
min_rmse = 1e+5

# 遍历特征

train_seq, train_label, val_seq, val_label, test_seq, test_label = split_dataset(df,window_size)
    # 遍历模型
for i in range(3):
    if i == 0:

        model = LSTMPredictor(train_seq.shape[2], 108, train_label.shape[1]).to(device)
    elif i == 1:
        model = BiLSTMPredictor(train_seq.shape[2], 108, train_label.shape[1]).to(device)
    else:
        model = GRUPredictor(train_seq.shape[2], 108, train_label.shape[1]).to(device)
    # print(f'current features: {features}', flush=True)
    print(f'current model_type: {type(model)}', flush=True)

        # 训练
    # if 'speeds' in features:
    #     model, loss, rmse = trainModel(train_seq, train_label, val_seq, val_label, model,
    #                                        lr, epoch_num * 2, logging_steps)
    # else:
    model, loss, rmse = trainModel(train_seq, train_label, val_seq, val_label, model,
                                           lr, epoch_num, logging_steps)
        # 在（划分的）测试集上测试
    predictions = []
    test_set = TensorDataset(test_seq, test_label)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, generator=torch.Generator(device=device))

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            predictions.append(outputs)

    predictions = torch.cat(predictions, dim=0)
    test_rmse = calc_rmse(predictions, test_label)
    print(f'test rmse: {test_rmse:.5f}')
    print(f'=' * 80, end='\n\n', flush=True)

        # 记录最好的模型
    if test_rmse < min_rmse:
        min_rmse = test_rmse
        best_model = model


# %%

print(f'best model: {best_model}')

print(f'min_rmse: {min_rmse:.6f}')



