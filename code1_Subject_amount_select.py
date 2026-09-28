import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
import re

# ----------------------
# ShallowCNN 模型
# ----------------------
class ShallowCNN(nn.Module):
    def __init__(self, num_classes, input_shape, dropout_rate=0.5):
        super().__init__()
        in_channels = input_shape[0]

        self.conv_time = nn.Conv2d(1, 40, kernel_size=(1,13), bias=False)
        self.conv_spat = nn.Conv2d(40, 40, kernel_size=(in_channels,1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.activation = nn.ELU(inplace=True)
        self.pool = nn.AvgPool2d((1,35), stride=(1,7))
        self.dropout = nn.Dropout(dropout_rate)

        with torch.no_grad():
            dummy = torch.randn(1, *input_shape)
            out = self._forward_features(dummy)
            self.fc_in = out.numel()

        self.fc = nn.Linear(self.fc_in, num_classes)

    def _forward_features(self, x):
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

# ----------------------
# 读取单个被试
# ----------------------
def load_subject(file_path):
    file_name = os.path.basename(file_path)
    subj_id = int(re.findall(r'\d+', file_name)[0])
    mat = loadmat(file_path)
    data = mat['data']
    labels = mat['label'].squeeze()
    if labels.ndim == 0:
        labels = np.array([labels.item()])
    if data.shape[-1] == len(labels):
        data = data.transpose(2,0,1)
    return data, labels, subj_id

# ----------------------
# 数据预处理
# ----------------------
def preprocess(source_data_list, target_data, target_labels, device):
    print("【步骤1】开始数据预处理：拼接源域所有被试数据")
    source_X = np.concatenate([d['data'] for d in source_data_list], axis=0)
    source_y = np.concatenate([d['labels'] for d in source_data_list], axis=0)
    print(f"源域数据shape: {source_X.shape}, 源域标签shape:{source_y.shape}")

    source_X = source_X[:, np.newaxis, :, :]
    target_data = target_data[:, np.newaxis, :, :]
    print("【步骤2】添加通道维度，开始标准化")

    mean = source_X.mean((0,2,3), keepdims=True) + 1e-8
    std  = source_X.std((0,2,3), keepdims=True) + 1e-8
    source_X = (source_X - mean) / std
    target_data = (target_data - mean) / std
    print("【步骤3】数据标准化完成，转换为GPU张量")

    X_source = torch.FloatTensor(source_X).to(device)
    y_source = torch.LongTensor(source_y).to(device)
    X_target = torch.FloatTensor(target_data).to(device)
    y_target = torch.LongTensor(target_labels).to(device)

    unique_labels = torch.unique(torch.cat([y_source, y_target])).cpu().numpy()
    label_map = {old:i for i,old in enumerate(unique_labels)}
    y_source = torch.LongTensor([label_map[l.item()] for l in y_source]).to(device)
    y_target = torch.LongTensor([label_map[l.item()] for l in y_target]).to(device)
    print(f"【步骤4】标签重映射完成，总类别数：{len(unique_labels)}")

    return X_source, y_source, X_target, y_target, source_X.shape[1:], len(unique_labels)

# ----------------------
# 训练 + 测试
# ----------------------
def train_test(Xs, ys, Xt, yt, input_shape, num_classes, device):
    print("【步骤5】初始化ShallowCNN模型、优化器与损失函数")
    model = ShallowCNN(num_classes, input_shape).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 训练
    model.train()
    loader = DataLoader(TensorDataset(Xs, ys), 32, shuffle=True)
    print("【步骤6】开始模型训练，共30轮epoch")
    for epoch in range(30):
        total_loss = 0
        for data, label in loader:
            optimizer.zero_grad()
            loss = criterion(model(data), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch+1}/30, 本轮平均loss: {total_loss/len(loader):.4f}")

    # 测试
    print("【步骤7】训练结束，进入评估模式，在目标被试数据上测试")
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        pred = model(Xt).argmax(1)
        preds = pred.cpu().numpy()
        trues = yt.cpu().numpy()
    print("【步骤8】预测完成，计算准确率")
    return accuracy_score(trues, preds)

# ----------------------
# 主程序：极简版
# ----------------------
if __name__ == "__main__":
    print("===== 跨被试脑电解码程序启动 =====")
    data_folder = r"E:\MATB36-source\MATB_36_epoch2"
    TEST_SUBJECT = 8  # 测试第8号被试
    print(f"【主程序】设置目标被试编号 TEST_SUBJECT = {TEST_SUBJECT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"【主程序】使用设备：{device}")

    # 加载所有被试
    print("【主程序】开始遍历文件夹读取所有mat被试文件")
    subjects = {}
    for f in glob.glob(os.path.join(data_folder, "*.mat")):
        try:
            data, labels, sid = load_subject(f)
            subjects[sid] = {"data": data, "labels": labels}
            print(f"    成功加载被试 {sid}")
        except:
            continue
    print(f"【主程序】全部加载完毕，共读取 {len(subjects)} 个被试")

    # 训练集：除目标被试外全部
    source_list = [subjects[sid] for sid in subjects if sid != TEST_SUBJECT]
    target_data = subjects[TEST_SUBJECT]["data"]
    target_labels = subjects[TEST_SUBJECT]["labels"]
    print(f"【主程序】源域被试数量：{len(source_list)}，目标被试{TEST_SUBJECT}数据已取出")

    # 预处理
    Xs, ys, Xt, yt, in_shape, n_class = preprocess(source_list, target_data, target_labels, device)

    # 训练 + 测试
    acc = train_test(Xs, ys, Xt, yt, in_shape, n_class, device)

    # 只输出结果
    print(f"\n===== 最终结果 =====")
    print(f"第{TEST_SUBJECT}号被试测试准确率：{acc:.4f}")
