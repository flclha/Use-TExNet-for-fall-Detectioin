import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import torch.optim as optim
from IRM import irm_loss
from sklearn.preprocessing import StandardScaler
from functions2 import self_sliding, lr_scheduler, CNN, Transformer, compute_accuracy, dynamic_weight_adjustment
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score, roc_auc_score, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

# 预训练网络自身应该泛化能力较弱，要证明其在目标数据上的泛化效果
# 公开数据集只训练，微调：训练自建数据集并测试、验证
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = 'cpu'

# ==================== 加载数据 ====================
data = pd.read_excel('Upfall4完整版数据(标注01）.xlsx')  # 引入处理数据文件

X = data.iloc[:, 1:13].values
y = data.iloc[:, 13].values
env_labels = data.iloc[:, 14].values  # 获取环境标签列（1~17）

# ==================== 数据预处理 ====================
scaler = StandardScaler()  # 归一化处理
X = scaler.fit_transform(X)

# ==================== 划分滑动窗口 ====================
X_sliding = self_sliding(X, window_width=200, stride=100)       # [num_samples, num_windows, 12]
y_sliding = self_sliding(y, window_width=200, stride=100)       # [num_samples, num_windows]
env_sliding = self_sliding(env_labels, window_width=200, stride=100)

num_samples = X_sliding.shape[0]
num_windows = X_sliding.shape[1]

# 展平为 [num_samples * num_windows, feature_dim]
X_combined = np.reshape(X_sliding, (num_samples * num_windows, 12))
y_combined = np.reshape(y_sliding, (num_samples * num_windows,))
env_combined = np.reshape(env_sliding, (num_samples * num_windows,))
env_combined = env_combined.astype(int)

print("总样本数（滑窗后）:", X_combined.shape[0])

# ==================== 按 env_label 划分训练 / 测试集 ====================
# env_label 1-12: 训练集；13-17: 测试集
train_mask = np.isin(env_combined, np.arange(1, 13))   # [1,2,...,12]
test_mask  = np.isin(env_combined, np.arange(13, 18))  # [13,14,15,16,17]

X_train = X_combined[train_mask]
y_train = y_combined[train_mask]
env_train = env_combined[train_mask]

X_test = X_combined[test_mask]
y_test = y_combined[test_mask]
env_test = env_combined[test_mask]

print("训练集样本数:", X_train.shape[0])
print("测试集样本数:", X_test.shape[0])

# ==================== 超参数 ====================
batch_size = 128
num_epoch = 65
weight_1 = 0.95  # 主要用于网络内部的 weight_1，可根据需要使用

# 判定为正类(1)的阈值：降低这个值，一般召回率会升、准确率可能下降
threshold = 0.3

# ==================== DataLoader ====================
train_dataset = torch.utils.data.TensorDataset(
    torch.Tensor(X_train),
    torch.Tensor(y_train),
    torch.Tensor(env_train)
)
test_dataset = torch.utils.data.TensorDataset(
    torch.Tensor(X_test),
    torch.Tensor(y_test),
    torch.Tensor(env_test)
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ==================== 模型定义（只在训练阶段更新动态权重 + Dropout） ====================
class MyNet(nn.Module):
    def __init__(self, input_dim, output_dim, weight_1):
        super(MyNet, self).__init__()
        self.cnn = CNN(input_dim, output_dim)
        self.transformer = Transformer(input_dim, output_dim, nhead=12)
        self.weight_cnn = nn.Parameter(torch.ones(1))
        self.weight_transformer = nn.Parameter(torch.ones(1))

        self.dropout = nn.Dropout(p=0.5)  # 抑制过拟合
        self.classifier = nn.Linear(output_dim, output_dim)
        self.weight_1 = weight_1

    def forward(self, x, ground_truth_labels=None):
        # x: [B, D]
        cnn_output = self.cnn(x)                  # [B, C]
        transformer_output = self.transformer(x)  # [B, C]

        # 只在训练阶段、且提供了标签时才根据准确率动态调整权重
        if self.training and (ground_truth_labels is not None):
            cnn_accuracy = compute_accuracy(cnn_output, ground_truth_labels)
            transformer_accuracy = compute_accuracy(transformer_output, ground_truth_labels)

            self.weight_cnn.data = dynamic_weight_adjustment(self.weight_cnn, cnn_accuracy)
            self.weight_transformer.data = dynamic_weight_adjustment(self.weight_transformer, transformer_accuracy)

        # 按权重加权融合
        weighted_output = self.weight_cnn * cnn_output + self.weight_transformer * transformer_output
        weighted_output = self.dropout(weighted_output)  # Dropout
        output = self.classifier(weighted_output)  # [B, 2]

        return output


large_model = MyNet(input_dim=X_train.shape[1], output_dim=2, weight_1=weight_1).to(device)

# ==================== 类别加权的交叉熵（提高少数类的权重） ====================
# 假设 y_train 为 0/1，统计各类别数量
class_counts = np.bincount(y_train.astype(int))
num_classes = len(class_counts)
total_samples = float(sum(class_counts))
class_weights = [total_samples / (num_classes * c) for c in class_counts]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Class counts:", class_counts)
print("Class weights used in CE:", class_weights)

# 稍微加大 weight_decay，进一步抑制过拟合
optimizer = optim.Adam(large_model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # 加权交叉熵

# ==================== 一些工具函数 ====================
def accuracy_score_tensor(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return (preds == labels).mean()


train_acc_list = []
train_loss_list = []

# 学习率调度
warmup_epochs = 5
decay_epochs = 80
initial_lr = 1e-4
base_lr = 1e-3
min_lr = 5e-5

# 用于保存指标
library = {
    'Epoch': [],
    'Train Loss': [],
    'Train Acc': [],
    'Train Recall': [],
    'Train F1': [],
    'Test Acc': [],
    'Test Recall': [],
    'Test F1': []
}

# ==================== Early Stopping 配置 ====================
best_test_f1 = 0.0
patience = 8
no_improve = 0

# ==================== 训练循环 ====================
for epoch in range(num_epoch):
    # ---- 更新学习率 ----
    lr = lr_scheduler(epoch, optimizer.param_groups[0]['lr'],
                      warmup_epochs, decay_epochs, initial_lr, base_lr, min_lr)
    optimizer.param_groups[0]['lr'] = lr

    # ---- 训练阶段 ----
    large_model.train()
    epoch_losses = []

    for inputs, labels, groups in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        groups = groups.to(device)

        optimizer.zero_grad()
        # 训练阶段传入 labels，用于动态权重调整
        outputs = large_model(inputs, labels.long())  # [B, 2]

        loss = criterion(outputs, labels.long())
        # 如果要用 IRM 替换：
        # loss = irm_loss(outputs, labels, groups)

        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    avg_train_loss = np.mean(epoch_losses)

    # ---- 评估阶段 ----
    large_model.eval()
    with torch.no_grad():
        # ========== 训练集指标 ==========
        train_preds = []
        train_trues = []

        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 评估阶段不再传 labels，避免动态权重在测试/验证中被篡改
            outputs = large_model(inputs, None)  # [B, 2]

            probs_1 = F.softmax(outputs, dim=1)[:, 1]      # 正类(1)概率
            preds = (probs_1 >= threshold).long()          # 阈值判定为 1/0

            train_preds.extend(preds.cpu().numpy().tolist())
            train_trues.extend(labels.cpu().numpy().tolist())

        train_acc = accuracy_score(train_trues, train_preds)
        train_recall = recall_score(train_trues, train_preds, zero_division=0)
        train_f1 = f1_score(train_trues, train_preds, zero_division=0)

        # ========== 测试集指标 ==========
        test_preds = []
        test_trues = []

        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = large_model(inputs, None)

            probs_1 = F.softmax(outputs, dim=1)[:, 1]
            preds = (probs_1 >= threshold).long()

            test_preds.extend(preds.cpu().numpy().tolist())
            test_trues.extend(labels.cpu().numpy().tolist())

        test_acc = accuracy_score(test_trues, test_preds)
        test_recall = recall_score(test_trues, test_preds, zero_division=0)
        test_f1 = f1_score(test_trues, test_preds, zero_division=0)

    # 记录
    train_acc_list.append(train_acc)
    train_loss_list.append(avg_train_loss)

    library['Epoch'].append(epoch + 1)
    library['Train Loss'].append(avg_train_loss)
    library['Train Acc'].append(train_acc)
    library['Train Recall'].append(train_recall)
    library['Train F1'].append(train_f1)
    library['Test Acc'].append(test_acc)
    library['Test Recall'].append(test_recall)
    library['Test F1'].append(test_f1)

    # ---- 打印该轮结果 ----
    print(f"Epoch {epoch + 1}/{num_epoch} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f} | "
          f"Test Acc: {test_acc:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    # ---- Early Stopping ----
    if test_f1 > best_test_f1:
        best_test_f1 = test_f1
        no_improve = 0
        # 保存当前最优模型
        torch.save(large_model.state_dict(), 'fine_tune_best.pth')
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}, best Test F1 = {best_test_f1:.4f}")
            break

# ==================== 保存结果到 Excel ====================
df = pd.DataFrame(library)
df.to_excel('evolution_env_split.xlsx', index=False)

# ==================== 画图 ====================
plt.figure(num='figure1', figsize=(10, 5))
plt.plot(range(1, len(train_acc_list) + 1), train_acc_list, label='Train Accuracy')
plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Accuracy and Loss')
plt.legend()
plt.show()

# ==================== 保存最后一轮模型（可选） ====================
torch.save(large_model.state_dict(), 'fine_tune_last.pth')
print("✅ 训练完成，最优模型已保存为 fine_tune_best.pth，训练过程已保存到 evolution_env_split.xlsx")
