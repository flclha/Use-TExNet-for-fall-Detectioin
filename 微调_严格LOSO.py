import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functions2 import lr_scheduler, CNN, Transformer, dynamic_weight_adjustment, compute_accuracy
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import pandas as pd
import os
import torch.nn.functional as F

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 固定随机种子，便于复现
np.random.seed(42)
torch.manual_seed(42)

# ==================== 基本设置 ====================
device = 'cpu'
num_epochs_finetune = 50   # 不要太大，配合早停即可
batch_size = 128

# 从测试 subject 抽多少样本混入训练（few-shot 比例）
# 严格 LOSO 时设置为 0
support_ratio = 0.0

# 学习率调度参数
warmup_epochs = 5
decay_epochs = 20
initial_lr = 1e-4
base_lr = 1e-3
min_lr = 5e-5

# ==================== 加载第二个数据集（原始，不在这里归一化） ====================
new_data = pd.read_excel('增强后的数据.xlsx')  # 引入新的数据文件

# 注意：这里不要做全局的 scaler.fit_transform
new_X_raw = new_data.iloc[:, 1:13].values.astype(np.float32)  # 原始特征
new_y = new_data.iloc[:, 13].values.astype(np.int64)          # 标签
new_env = new_data.iloc[:, 14].values.astype(np.int64)        # 用于 LOSO 的 env / subject 标签

# ==================== 定义模型 ====================
class MyNet(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(MyNet, self).__init__()
        self.cnn = CNN(input_dim, output_dim)
        self.transformer = Transformer(input_dim, output_dim, nhead=12)
        self.weight_cnn = nn.Parameter(torch.ones(1))
        self.weight_transformer = nn.Parameter(torch.ones(1))
        self.classifier = nn.Linear(output_dim, output_dim)

    def forward(self, x, ground_truth_labels):
        # x: [B, ...], ground_truth_labels: [B]
        cnn_output = self.cnn(x)
        transformer_output = self.transformer(x)

        # 根据当前 batch 的准确率动态调整权重
        cnn_accuracy = compute_accuracy(cnn_output, ground_truth_labels)
        transformer_accuracy = compute_accuracy(transformer_output, ground_truth_labels)

        self.weight_cnn.data = dynamic_weight_adjustment(self.weight_cnn, cnn_accuracy)
        self.weight_transformer.data = dynamic_weight_adjustment(self.weight_transformer, transformer_accuracy)

        weighted_output = self.weight_cnn * cnn_output + self.weight_transformer * transformer_output
        output = self.classifier(weighted_output)  # [B, 2]
        return output

# ==================== LOSO 主循环：按 new_env 留一 ====================
unique_envs = np.unique(new_env)
print("所有 env ID:", unique_envs)

fold_results = []

for test_env in unique_envs:
    print(f"\n===== 开始 LOSO+few-shot：env = {test_env} 作为测试 subject =====")

    # ------- 按 env 划分最外层 train / test ------- #
    outer_train_mask = new_env != test_env
    outer_test_mask  = new_env == test_env

    if outer_test_mask.sum() == 0:
        print(f"env = {test_env} 没有样本，跳过该 fold")
        continue

    # 注意：这里用的是“原始特征”，尚未归一化
    X_train_outer_raw = new_X_raw[outer_train_mask]
    y_train_outer = new_y[outer_train_mask]
    env_train_outer = new_env[outer_train_mask]

    X_test_raw = new_X_raw[outer_test_mask]
    y_test_fold = new_y[outer_test_mask]
    env_test_fold = new_env[outer_test_mask]

    n_test_all = X_test_raw.shape[0]
    print(f"Fold env={test_env}: 原始训练样本 {X_train_outer_raw.shape[0]}, 原始测试样本 {n_test_all}")

    # ========== few-shot 逻辑（严格 LOSO 时不混入任何测试样本） ==========
    if support_ratio > 0.0 and n_test_all > 3:
        # 只有 support_ratio > 0 时才允许从测试集抽样
        n_support = int(n_test_all * support_ratio)

        if n_support > 0:
            support_indices = np.random.choice(n_test_all, size=n_support, replace=False)

            X_support_raw = X_test_raw[support_indices]
            y_support = y_test_fold[support_indices]
            env_support = env_test_fold[support_indices]

            # 测试集中剩余样本
            keep_mask = np.ones(n_test_all, dtype=bool)
            keep_mask[support_indices] = False

            X_test_raw_final = X_test_raw[keep_mask]
            y_test_fold_final = y_test_fold[keep_mask]
            env_test_fold_final = env_test_fold[keep_mask]

            # 把 support 样本拼到训练集中
            X_train_outer_raw = np.concatenate([X_train_outer_raw, X_support_raw], axis=0)
            y_train_outer = np.concatenate([y_train_outer, y_support], axis=0)
            env_train_outer = np.concatenate([env_train_outer, env_support], axis=0)

            print(f"从 env={test_env} 测试集中随机抽取 {n_support} 个样本加入训练集，"
                  f"剩余 {X_test_raw_final.shape[0]} 个样本作为测试集")
        else:
            # support_ratio 很小导致 n_support=0，等价于不做 few-shot
            X_test_raw_final = X_test_raw
            y_test_fold_final = y_test_fold
            env_test_fold_final = env_test_fold
            print(f"support_ratio 太小导致 n_support=0，本 fold 不做 few-shot 混入")
    else:
        # 严格 LOSO 或测试样本太少：完全不从测试集中抽样
        X_test_raw_final = X_test_raw
        y_test_fold_final = y_test_fold
        env_test_fold_final = env_test_fold
        if support_ratio == 0.0:
            print(f"严格 LOSO 模式（support_ratio=0），不从 env={test_env} 的测试集中抽样到训练集")
        else:
            print(f"env={test_env} 测试样本过少（{n_test_all}），不做 support 抽样混入训练集")

    # 用“final”版本覆盖后续变量，避免到处改名
    X_test_raw = X_test_raw_final
    y_test_fold = y_test_fold_final
    env_test_fold = env_test_fold_final

    if X_test_raw.shape[0] == 0:
        print(f"env={test_env} 在 few-shot 抽样后测试集为空，跳过该 fold")
        continue

    print(f"Fold env={test_env}: 最终训练样本 {X_train_outer_raw.shape[0]}, 最终测试样本 {X_test_raw.shape[0]}")

    # ------- 在 outer_train 里再划分出 inner train / val（用于早停） ------- #
    X_train_inner_raw, X_val_inner_raw, y_train_inner, y_val_inner = train_test_split(
        X_train_outer_raw,
        y_train_outer,
        test_size=0.2,
        random_state=42,
        stratify=y_train_outer
    )

    # ------- 每个 fold 单独拟合 StandardScaler（只用 inner train） ------- #
    scaler = StandardScaler()
    scaler.fit(X_train_inner_raw)   # 只在 inner train 上拟合

    X_train_inner = scaler.transform(X_train_inner_raw).astype(np.float32)
    X_val_inner   = scaler.transform(X_val_inner_raw).astype(np.float32)
    X_test_fold_scaled = scaler.transform(X_test_raw).astype(np.float32)

    # ------- 构建 DataLoader ------- #
    train_data = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train_inner),
        torch.from_numpy(y_train_inner),
        torch.zeros_like(torch.from_numpy(y_train_inner))  # groups 占位，不用 IRM 也要一个 tensor
    )
    val_data = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val_inner),
        torch.from_numpy(y_val_inner),
        torch.zeros_like(torch.from_numpy(y_val_inner))
    )
    test_data = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test_fold_scaled),
        torch.from_numpy(y_test_fold),
        torch.from_numpy(env_test_fold)   # 这里 env 只是占位，目前没用到
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_data,   batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    # ------- 初始化模型并加载预训练参数（每个 fold 单独微调） ------- #
    model2 = MyNet(input_dim=12, output_dim=2, rank=4).to(device)
    state_dict = torch.load('fine_tune.pth', map_location=device)
    model2.load_state_dict(state_dict)

    # 先冻结 CNN + Transformer，只训练分类层
    for p in model2.cnn.parameters():
        p.requires_grad = False
    for p in model2.transformer.parameters():
        p.requires_grad = False

    # 计算 inner train 的类别权重（适度偏向正类）
    y_train_int = y_train_inner.astype(int)
    num_pos = (y_train_int == 1).sum()
    num_neg = (y_train_int == 0).sum()
    if num_pos == 0:
        pos_weight = 1.0
        neg_weight = 1.0
    else:
        total = num_pos + num_neg
        # 按频率反比设置权重，避免极端不平衡
        neg_weight = total / (2.0 * max(num_neg, 1))
        pos_weight = total / (2.0 * max(num_pos, 1))

    class_weights = torch.tensor([neg_weight, pos_weight], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 初始优化器：只训分类器
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model2.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    # 早停相关：在 val 上用 F2 分数（更重视 Recall）选最佳 epoch
    best_val_f2 = -1.0
    best_state_dict = None
    beta = 2.0

    for epoch in range(num_epochs_finetune):
        # 第 10 轮后解冻 backbone，一起细调
        if epoch == 10:
            for p in model2.cnn.parameters():
                p.requires_grad = True
            for p in model2.transformer.parameters():
                p.requires_grad = True

            optimizer = optim.Adam([
                {'params': model2.cnn.parameters(),         'lr': 1e-4},
                {'params': model2.transformer.parameters(), 'lr': 1e-4},
                {'params': model2.classifier.parameters(),  'lr': 1e-3},
            ], weight_decay=1e-4)

        # ---- 更新学习率（这里用第一个 param_group 的 lr 做基准）---- #
        lr = lr_scheduler(
            epoch,
            optimizer.param_groups[0]['lr'],
            warmup_epochs, decay_epochs,
            initial_lr, base_lr, min_lr
        )
        for g in optimizer.param_groups:
            g['lr'] = lr

        # ---- 训练阶段（inner train）---- #
        model2.train()
        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model2(inputs, labels)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

        # ---- 评估阶段：在 inner train & val & test 上看指标 ---- #
        model2.eval()
        with torch.no_grad():
            # ===== 训练集指标（看收敛情况）===== #
            train_preds_all = []
            train_trues_all = []
            for inputs, labels, _ in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model2(inputs, labels)
                preds = torch.argmax(outputs, dim=1)

                train_preds_all.extend(preds.cpu().numpy().tolist())
                train_trues_all.extend(labels.cpu().numpy().tolist())

            train_acc = accuracy_score(train_trues_all, train_preds_all)
            train_rec = recall_score(train_trues_all, train_preds_all, zero_division=0)
            train_f1 = f1_score(train_trues_all, train_preds_all, zero_division=0)

            # ===== 验证集指标（用于早停 + 选最佳 epoch）===== #
            val_preds_all = []
            val_trues_all = []
            for inputs, labels, _ in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model2(inputs, labels)
                preds = torch.argmax(outputs, dim=1)

                val_preds_all.extend(preds.cpu().numpy().tolist())
                val_trues_all.extend(labels.cpu().numpy().tolist())

            val_acc = accuracy_score(val_trues_all, val_preds_all)
            val_rec = recall_score(val_trues_all, val_preds_all, zero_division=0)
            val_prec = precision_score(val_trues_all, val_preds_all, zero_division=0)
            if (val_prec + val_rec) > 0:
                val_f2 = (1 + beta**2) * val_prec * val_rec / (beta**2 * val_prec + val_rec)
            else:
                val_f2 = 0.0

            # ===== 测试集指标（每一轮都看一下，仅做观察，不参与早停）===== #
            test_epoch_preds = []
            test_epoch_trues = []
            for inputs, labels, _ in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model2(inputs, labels)
                preds = torch.argmax(outputs, dim=1)

                test_epoch_preds.extend(preds.cpu().numpy().tolist())
                test_epoch_trues.extend(labels.cpu().numpy().tolist())

            test_epoch_acc = accuracy_score(test_epoch_trues, test_epoch_preds)
            test_epoch_rec = recall_score(test_epoch_trues, test_epoch_preds, zero_division=0)

            # 选择“验证集 F2 分数最优”的那一轮作为最佳模型
            if val_f2 > best_val_f2:
                best_val_f2 = val_f2
                best_state_dict = {k: v.cpu().clone() for k, v in model2.state_dict().items()}

        # 打印当前 epoch 的训练 + 验证 + 测试信息
        print(
            f"[Env {test_env}] Epoch {epoch + 1}/{num_epochs_finetune} | "
            f"Train Acc: {train_acc:.4f}, Train Rec: {train_rec:.4f}, Train F1: {train_f1:.4f} | "
            f"Val Acc: {val_acc:.4f}, Val Rec: {val_rec:.4f}, Val F2: {val_f2:.4f} | "
            f"Test Acc: {test_epoch_acc:.4f}, Test Rec: {test_epoch_rec:.4f}"
        )

    # ===== 一个 fold 训练完毕：用“验证集 F2 最优的参数”在测试集上做最终评估 ===== #
    if best_state_dict is not None:
        model2.load_state_dict(best_state_dict)

    model2.eval()
    with torch.no_grad():
        test_preds_all = []
        test_trues_all = []

        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model2(inputs, labels)
            preds = torch.argmax(outputs, dim=1)

            test_preds_all.extend(preds.cpu().numpy().tolist())
            test_trues_all.extend(labels.cpu().numpy().tolist())

        test_acc = accuracy_score(test_trues_all, test_preds_all)
        test_rec = recall_score(test_trues_all, test_preds_all, zero_division=0)
        test_f1 = f1_score(test_trues_all, test_preds_all, zero_division=0)

    print(f"\n[Env {test_env}] 使用验证集 F2 最优模型在测试集上的最终结果："
          f"Test Acc: {test_acc:.4f}, Test Rec: {test_rec:.4f}, Test F1: {test_f1:.4f}\n")

    fold_results.append({
        'env': int(test_env),
        'test_acc': test_acc,
        'test_rec': test_rec,
        'test_f1': test_f1
    })

# ==================== 汇总所有 env 的测试结果 ====================
results_df = pd.DataFrame(fold_results)
print("\n===== 各 env 测试结果 =====")
print(results_df)

print("\n===== 平均测试性能 =====")
print("Mean Acc:", results_df['test_acc'].mean())
print("Mean Rec:", results_df['test_rec'].mean())
print("Mean F1 :", results_df['test_f1'].mean())

results_df.to_excel('loso_finetune_test_results_fewshot.xlsx', index=False)
print("\n✅ LOSO + few-shot 微调完成，所有 env 的测试结果已保存到 loso_finetune_test_results_fewshot.xlsx")
