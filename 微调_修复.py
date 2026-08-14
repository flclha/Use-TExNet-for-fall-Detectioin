import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import torch.optim as optim
# 预训练参数使用的固定权重，微调阶段仍动态调整cnn transformer权重
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve
)

from functions2 import (
    lr_scheduler,
    CNN,
    Transformer,
    dynamic_weight_adjustment,
    compute_accuracy
)

import pandas as pd
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Basic settings
# ============================================================

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# ------------------------------------------------------------
# Fixed random seed
# ------------------------------------------------------------

np.random.seed(42)

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ============================================================
# Fine-tuning parameters
# ============================================================

num_epochs_finetune = 80

batch_size = 128


# ------------------------------------------------------------
# Few-shot ratio
#
# 从目标 subject 中抽取 10% 样本加入训练
# ------------------------------------------------------------

support_ratio = 0.10


# ------------------------------------------------------------
# Few-shot sampling mode
#
# random:
# ADL + Fall 都可以抽
#
# ADL_only:
# 只从 ADL 中抽
# ------------------------------------------------------------

support_mode = "random"


# ============================================================
# Learning-rate parameters
# ============================================================

warmup_epochs = 5

decay_epochs = 20

initial_lr = 1e-4

base_lr = 1e-3

min_lr = 5e-5


# ============================================================
# Load dataset
# ============================================================

new_data = pd.read_excel(
    "增强后的数据.xlsx"
)


# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

new_X_raw = new_data.iloc[
    :,
    1:13
].values.astype(
    np.float32
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

new_y = new_data.iloc[
    :,
    13
].values.astype(
    np.int64
)


# ------------------------------------------------------------
# Subject / environment ID
# ------------------------------------------------------------

new_env = new_data.iloc[
    :,
    14
].values.astype(
    np.int64
)


print(
    "所有 env ID:",
    np.unique(new_env)
)


# ============================================================
# Model
# ============================================================

class MyNet(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        rank=4
    ):

        super(
            MyNet,
            self
        ).__init__()


        # ====================================================
        # CNN branch
        # ====================================================

        self.cnn = CNN(
            input_dim,
            output_dim
        )


        # ====================================================
        # Transformer branch
        # ====================================================

        self.transformer = Transformer(
            input_dim,
            output_dim,
            nhead=12
        )


        # ====================================================
        # Dynamic fusion weights
        #
        # 仍然放进 state_dict，
        # 可以从预训练 pth 中加载。
        #
        # 但这里不让 optimizer 再更新，
        # 只通过 dynamic_weight_adjustment 更新。
        # ====================================================

        self.weight_cnn = nn.Parameter(
            torch.ones(1),
            requires_grad=False
        )

        self.weight_transformer = nn.Parameter(
            torch.ones(1),
            requires_grad=False
        )


        # ====================================================
        # Classifier
        # ====================================================

        self.classifier = nn.Linear(
            output_dim,
            output_dim
        )


    def forward(
        self,
        x,
        ground_truth_labels=None,
        update_dynamic_weights=False
    ):

        # ====================================================
        # CNN
        # ====================================================

        cnn_output = self.cnn(
            x
        )


        # ====================================================
        # Transformer
        # ====================================================

        transformer_output = self.transformer(
            x
        )


        # ====================================================
        # Dynamic-weight update
        #
        # 只有训练阶段才允许执行
        # ====================================================

        if (
            self.training
            and
            update_dynamic_weights
            and
            ground_truth_labels is not None
        ):

            # ------------------------------------------------
            # Calculate branch accuracy
            # ------------------------------------------------

            cnn_accuracy = compute_accuracy(
                cnn_output,
                ground_truth_labels
            )

            transformer_accuracy = compute_accuracy(
                transformer_output,
                ground_truth_labels
            )


            # ------------------------------------------------
            # Adjust CNN weight
            # ------------------------------------------------

            new_weight_cnn = dynamic_weight_adjustment(
                self.weight_cnn,
                cnn_accuracy
            )


            # ------------------------------------------------
            # Adjust Transformer weight
            # ------------------------------------------------

            new_weight_transformer = (
                dynamic_weight_adjustment(
                    self.weight_transformer,
                    transformer_accuracy
                )
            )


            # ------------------------------------------------
            # Update weights without gradient
            # ------------------------------------------------

            with torch.no_grad():

                self.weight_cnn.copy_(
                    new_weight_cnn
                )

                self.weight_transformer.copy_(
                    new_weight_transformer
                )


        # ====================================================
        # Dynamic fusion
        # ====================================================

        weighted_output = (

            self.weight_cnn
            *
            cnn_output

            +

            self.weight_transformer
            *
            transformer_output
        )


        # ====================================================
        # Classification
        # ====================================================

        output = self.classifier(
            weighted_output
        )


        return output


# ============================================================
# LOSO
# ============================================================

unique_envs = np.unique(
    new_env
)

fold_results = []


# ============================================================
# Overall ROC storage
# ============================================================

all_test_labels_global = []

all_test_scores_global = []


# ============================================================
# LOSO main loop
# ============================================================

for test_env in unique_envs:

    print(
        "\n"
        "======================================================"
    )

    print(
        f"开始 LOSO + few-shot："
        f"env = {test_env} 作为测试 subject"
    )

    print(
        "======================================================"
    )


    # ========================================================
    # Outer LOSO split
    # ========================================================

    outer_train_mask = (
        new_env
        !=
        test_env
    )

    outer_test_mask = (
        new_env
        ==
        test_env
    )


    # --------------------------------------------------------
    # Source subjects
    # --------------------------------------------------------

    X_train_outer_raw = new_X_raw[
        outer_train_mask
    ]

    y_train_outer = new_y[
        outer_train_mask
    ]

    env_train_outer = new_env[
        outer_train_mask
    ]


    # --------------------------------------------------------
    # Target subject
    # --------------------------------------------------------

    X_test_raw = new_X_raw[
        outer_test_mask
    ]

    y_test_fold = new_y[
        outer_test_mask
    ]

    env_test_fold = new_env[
        outer_test_mask
    ]


    n_test_all = (
        X_test_raw.shape[0]
    )


    print(
        f"Fold env={test_env}: "
        f"原始训练样本 "
        f"{X_train_outer_raw.shape[0]}, "
        f"原始测试样本 "
        f"{n_test_all}"
    )


    # ========================================================
    # Few-shot support sampling
    # ========================================================

    if n_test_all > 3:

        # ----------------------------------------------------
        # Candidate support indices
        # ----------------------------------------------------

        if support_mode == "ADL_only":

            candidate_indices = np.where(
                y_test_fold == 0
            )[0]

        else:

            candidate_indices = np.arange(
                n_test_all
            )


        # ----------------------------------------------------
        # Handle empty candidates
        # ----------------------------------------------------

        if len(candidate_indices) > 0:

            n_support = max(
                1,
                int(
                    len(candidate_indices)
                    *
                    support_ratio
                )
            )


            n_support = min(
                n_support,
                len(candidate_indices)
            )


            support_indices = np.random.choice(
                candidate_indices,
                size=n_support,
                replace=False
            )

        else:

            support_indices = np.array(
                [],
                dtype=int
            )


        # ----------------------------------------------------
        # Support samples
        # ----------------------------------------------------

        X_support_raw = X_test_raw[
            support_indices
        ]

        y_support = y_test_fold[
            support_indices
        ]

        env_support = env_test_fold[
            support_indices
        ]


        # ----------------------------------------------------
        # Remove support samples from final test set
        # ----------------------------------------------------

        keep_mask = np.ones(
            n_test_all,
            dtype=bool
        )

        keep_mask[
            support_indices
        ] = False


        X_test_raw_final = X_test_raw[
            keep_mask
        ]

        y_test_fold_final = y_test_fold[
            keep_mask
        ]

        env_test_fold_final = env_test_fold[
            keep_mask
        ]


        # ----------------------------------------------------
        # Add support into training
        # ----------------------------------------------------

        if len(
            support_indices
        ) > 0:

            X_train_outer_raw = np.concatenate(
                [
                    X_train_outer_raw,
                    X_support_raw
                ],
                axis=0
            )

            y_train_outer = np.concatenate(
                [
                    y_train_outer,
                    y_support
                ],
                axis=0
            )

            env_train_outer = np.concatenate(
                [
                    env_train_outer,
                    env_support
                ],
                axis=0
            )


        print(
            f"从 env={test_env} 中抽取 "
            f"{len(support_indices)} "
            f"个 few-shot support 样本加入训练集"
        )

        print(
            f"剩余 "
            f"{X_test_raw_final.shape[0]} "
            f"个样本作为测试集"
        )


    else:

        # ----------------------------------------------------
        # Too few target samples
        # ----------------------------------------------------

        X_support_raw = np.empty(
            (
                0,
                X_test_raw.shape[1]
            ),
            dtype=np.float32
        )

        y_support = np.empty(
            (0,),
            dtype=np.int64
        )

        env_support = np.empty(
            (0,),
            dtype=np.int64
        )


        X_test_raw_final = (
            X_test_raw
        )

        y_test_fold_final = (
            y_test_fold
        )

        env_test_fold_final = (
            env_test_fold
        )


        print(
            f"env={test_env} "
            f"测试样本过少，"
            f"不进行 few-shot support 抽样"
        )


    # ========================================================
    # Final test set
    # ========================================================

    X_test_raw = (
        X_test_raw_final
    )

    y_test_fold = (
        y_test_fold_final
    )

    env_test_fold = (
        env_test_fold_final
    )


    if (
        X_test_raw.shape[0]
        ==
        0
    ):

        print(
            f"env={test_env} "
            f"测试集为空，跳过"
        )

        continue


    print(
        f"Fold env={test_env}: "
        f"最终训练样本 "
        f"{X_train_outer_raw.shape[0]}, "
        f"最终测试样本 "
        f"{X_test_raw.shape[0]}"
    )


    # ========================================================
    # Normalization
    #
    # scaler 只 fit 训练数据
    # ========================================================

    scaler = StandardScaler()


    scaler.fit(
        X_train_outer_raw
    )


    X_train_inner = scaler.transform(
        X_train_outer_raw
    ).astype(
        np.float32
    )


    X_test_fold_scaled = scaler.transform(
        X_test_raw
    ).astype(
        np.float32
    )


    y_train_inner = (
        y_train_outer
    )


    # ========================================================
    # Dataset
    # ========================================================

    train_data = torch.utils.data.TensorDataset(

        torch.from_numpy(
            X_train_inner
        ),

        torch.from_numpy(
            y_train_inner.astype(
                np.int64
            )
        ),

        torch.from_numpy(
            env_train_outer.astype(
                np.int64
            )
        )
    )


    test_data = torch.utils.data.TensorDataset(

        torch.from_numpy(
            X_test_fold_scaled
        ),

        torch.from_numpy(
            y_test_fold.astype(
                np.int64
            )
        ),

        torch.from_numpy(
            env_test_fold.astype(
                np.int64
            )
        )
    )


    # ========================================================
    # DataLoader
    # ========================================================

    train_loader = torch.utils.data.DataLoader(

        train_data,

        batch_size=batch_size,

        shuffle=True
    )


    test_loader = torch.utils.data.DataLoader(

        test_data,

        batch_size=batch_size,

        shuffle=False
    )


    # ========================================================
    # Initialize model
    # ========================================================

    model2 = MyNet(

        input_dim=12,

        output_dim=2,

        rank=4

    ).to(
        device
    )


    # ========================================================
    # Load pretrained model
    # ========================================================

    state_dict = torch.load(

        "IRM_pretrained.pth",

        map_location=device
    )


    model2.load_state_dict(

        state_dict,

        strict=True
    )


    print(
        "预训练模型加载成功"
    )


    print(
        "Initial weight_cnn:",
        model2.weight_cnn.item()
    )

    print(
        "Initial weight_transformer:",
        model2.weight_transformer.item()
    )


    # ========================================================
    # Freeze CNN + Transformer first
    # ========================================================

    for p in model2.cnn.parameters():

        p.requires_grad = False


    for p in model2.transformer.parameters():

        p.requires_grad = False


    # ========================================================
    # Class imbalance weights
    # ========================================================

    y_train_int = y_train_inner.astype(
        int
    )


    num_pos = (
        y_train_int == 1
    ).sum()


    num_neg = (
        y_train_int == 0
    ).sum()


    if (
        num_pos == 0
        or
        num_neg == 0
    ):

        neg_weight = 1.0

        pos_weight = 1.0


    else:

        total = (
            num_pos
            +
            num_neg
        )


        neg_weight = (
            total
            /
            (
                2.0
                *
                num_neg
            )
        )


        pos_weight = (
            total
            /
            (
                2.0
                *
                num_pos
            )
        )


    class_weights = torch.tensor(

        [
            neg_weight,
            pos_weight
        ],

        dtype=torch.float32

    ).to(
        device
    )


    print(
        "Class weights:",
        class_weights
        .detach()
        .cpu()
        .numpy()
    )


    criterion = nn.CrossEntropyLoss(

        weight=class_weights
    )


    # ========================================================
    # Initial optimizer
    #
    # First 10 epochs:
    # classifier only
    # ========================================================

    optimizer = optim.Adam(

        model2.classifier.parameters(),

        lr=1e-3,

        weight_decay=1e-4
    )


    # ========================================================
    # History
    # ========================================================

    history_epoch = []

    history_train_loss = []

    history_train_acc = []

    history_train_rec = []

    history_train_f1 = []

    history_test_acc = []

    history_test_rec = []

    history_test_f1 = []

    history_weight_cnn = []

    history_weight_transformer = []


    # ========================================================
    # Fine-tuning
    # ========================================================

    for epoch in range(
        num_epochs_finetune
    ):


        # ====================================================
        # Epoch 11:
        # Unfreeze CNN + Transformer
        # ====================================================

        if epoch == 10:

            print(
                "\n"
                ">>> 解冻 CNN + Transformer "
                "开始联合微调"
            )


            for p in model2.cnn.parameters():

                p.requires_grad = True


            for p in model2.transformer.parameters():

                p.requires_grad = True


            optimizer = optim.Adam(

                [

                    {
                        "params":
                        model2.cnn.parameters(),

                        "lr":
                        1e-4
                    },

                    {
                        "params":
                        model2.transformer.parameters(),

                        "lr":
                        1e-4
                    },

                    {
                        "params":
                        model2.classifier.parameters(),

                        "lr":
                        1e-3
                    }

                ],

                weight_decay=1e-4
            )


        # ====================================================
        # Learning-rate scheduling
        # ====================================================

        lr = lr_scheduler(

            epoch,

            optimizer.param_groups[
                0
            ]["lr"],

            warmup_epochs,

            decay_epochs,

            initial_lr,

            base_lr,

            min_lr
        )


        for group in optimizer.param_groups:

            group["lr"] = lr


        # ====================================================
        # TRAINING
        #
        # Only training labels can update dynamic weights
        # ====================================================

        model2.train()


        train_loss_sum = 0.0

        train_sample_count = 0


        for (
            inputs,
            labels,
            _
        ) in train_loader:


            inputs = inputs.to(
                device
            )

            labels = labels.to(
                device
            )


            optimizer.zero_grad()


            # ------------------------------------------------
            # Training:
            #
            # labels are passed
            # dynamic weights are updated
            # ------------------------------------------------

            outputs = model2(

                inputs,

                ground_truth_labels=labels,

                update_dynamic_weights=True
            )


            loss = criterion(

                outputs,

                labels.long()
            )


            loss.backward()


            optimizer.step()


            current_batch_size = (
                labels.size(0)
            )


            train_loss_sum += (
                loss.item()
                *
                current_batch_size
            )


            train_sample_count += (
                current_batch_size
            )


        train_loss_epoch = (

            train_loss_sum
            /
            max(
                train_sample_count,
                1
            )
        )


        # ====================================================
        # TRAIN SET EVALUATION
        #
        # 不再更新 dynamic weights
        # ====================================================

        model2.eval()


        train_preds_all = []

        train_trues_all = []


        with torch.no_grad():

            for (
                inputs,
                labels,
                _
            ) in train_loader:


                inputs = inputs.to(
                    device
                )

                labels = labels.to(
                    device
                )


                # --------------------------------------------
                # Evaluation:
                # labels are NOT passed to forward
                # --------------------------------------------

                outputs = model2(
                    inputs
                )


                preds = torch.argmax(

                    outputs,

                    dim=1
                )


                train_preds_all.extend(

                    preds
                    .cpu()
                    .numpy()
                    .tolist()
                )


                train_trues_all.extend(

                    labels
                    .cpu()
                    .numpy()
                    .tolist()
                )


        # ----------------------------------------------------
        # Training metrics
        # ----------------------------------------------------

        train_acc = accuracy_score(

            train_trues_all,

            train_preds_all
        )


        train_rec = recall_score(

            train_trues_all,

            train_preds_all,

            zero_division=0
        )


        train_f1 = f1_score(

            train_trues_all,

            train_preds_all,

            zero_division=0
        )


        # ====================================================
        # TEST SET EVALUATION
        #
        # IMPORTANT:
        #
        # test labels NEVER enter forward()
        # ====================================================

        test_preds_epoch = []

        test_trues_epoch = []


        with torch.no_grad():

            for (
                inputs,
                labels,
                _
            ) in test_loader:


                inputs = inputs.to(
                    device
                )

                labels = labels.to(
                    device
                )


                # ============================================
                # CRITICAL:
                #
                # Correct:
                #
                # outputs = model2(inputs)
                #
                # Wrong:
                #
                # outputs = model2(inputs, labels)
                # ============================================

                outputs = model2(
                    inputs
                )


                preds = torch.argmax(

                    outputs,

                    dim=1
                )


                test_preds_epoch.extend(

                    preds
                    .cpu()
                    .numpy()
                    .tolist()
                )


                test_trues_epoch.extend(

                    labels
                    .cpu()
                    .numpy()
                    .tolist()
                )


        # ----------------------------------------------------
        # Test metrics
        # ----------------------------------------------------

        test_acc_epoch = accuracy_score(

            test_trues_epoch,

            test_preds_epoch
        )


        test_rec_epoch = recall_score(

            test_trues_epoch,

            test_preds_epoch,

            zero_division=0
        )


        test_f1_epoch = f1_score(

            test_trues_epoch,

            test_preds_epoch,

            zero_division=0
        )


        # ====================================================
        # Current dynamic weights
        # ====================================================

        current_weight_cnn = (
            model2
            .weight_cnn
            .detach()
            .cpu()
            .item()
        )


        current_weight_transformer = (
            model2
            .weight_transformer
            .detach()
            .cpu()
            .item()
        )


        # ====================================================
        # Save history
        # ====================================================

        history_epoch.append(
            epoch + 1
        )

        history_train_loss.append(
            train_loss_epoch
        )

        history_train_acc.append(
            train_acc
        )

        history_train_rec.append(
            train_rec
        )

        history_train_f1.append(
            train_f1
        )

        history_test_acc.append(
            test_acc_epoch
        )

        history_test_rec.append(
            test_rec_epoch
        )

        history_test_f1.append(
            test_f1_epoch
        )

        history_weight_cnn.append(
            current_weight_cnn
        )

        history_weight_transformer.append(
            current_weight_transformer
        )


        # ====================================================
        # Print every epoch
        # ====================================================

        print(

            f"[Env {test_env}] "

            f"Epoch "
            f"{epoch + 1:02d}/"
            f"{num_epochs_finetune} | "

            f"Loss:"
            f"{train_loss_epoch:.4f} | "

            f"Train Acc:"
            f"{train_acc:.4f}, "

            f"Train Rec:"
            f"{train_rec:.4f}, "

            f"Train F1:"
            f"{train_f1:.4f} | "

            f"Test Acc:"
            f"{test_acc_epoch:.4f}, "

            f"Test Rec:"
            f"{test_rec_epoch:.4f}, "

            f"Test F1:"
            f"{test_f1_epoch:.4f} | "

            f"W_CNN:"
            f"{current_weight_cnn:.4f}, "

            f"W_TRANS:"
            f"{current_weight_transformer:.4f}"
        )


    # ========================================================
    # Final evaluation after Epoch 50
    #
    # 不选择 Test 最优 epoch
    # 直接使用训练结束后的模型
    # ========================================================

    model2.eval()


    test_preds_all = []

    test_trues_all = []

    test_scores_all = []


    with torch.no_grad():

        for (
            inputs,
            labels,
            _
        ) in test_loader:


            inputs = inputs.to(
                device
            )

            labels = labels.to(
                device
            )


            # ------------------------------------------------
            # Test labels are NOT passed to model
            # ------------------------------------------------

            outputs = model2(
                inputs
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            preds = torch.argmax(

                outputs,

                dim=1
            )


            # ------------------------------------------------
            # Positive-class probability
            # ------------------------------------------------

            probs = F.softmax(

                outputs,

                dim=1

            )[:, 1]


            test_preds_all.extend(

                preds
                .cpu()
                .numpy()
                .tolist()
            )


            test_trues_all.extend(

                labels
                .cpu()
                .numpy()
                .tolist()
            )


            test_scores_all.extend(

                probs
                .cpu()
                .numpy()
                .tolist()
            )


    # ========================================================
    # Final metrics
    # ========================================================

    test_acc = accuracy_score(

        test_trues_all,

        test_preds_all
    )


    test_precision = precision_score(

        test_trues_all,

        test_preds_all,

        zero_division=0
    )


    test_rec = recall_score(

        test_trues_all,

        test_preds_all,

        zero_division=0
    )


    test_f1 = f1_score(

        test_trues_all,

        test_preds_all,

        zero_division=0
    )


    # ========================================================
    # AUC
    #
    # 防止某个 subject 只有一个类别
    # ========================================================

    if (
        len(
            np.unique(
                test_trues_all
            )
        )
        >=
        2
    ):

        test_auc = roc_auc_score(

            test_trues_all,

            test_scores_all
        )

    else:

        test_auc = np.nan


    # ========================================================
    # Print final fold result
    # ========================================================

    print(
        "\n"
        "------------------------------------------------------"
    )

    print(
        f"[Env {test_env}] "
        f"Epoch {num_epochs_finetune} "
        f"最终测试结果"
    )

    print(
        "------------------------------------------------------"
    )

    print(
        f"Test Accuracy  : "
        f"{test_acc:.4f}"
    )

    print(
        f"Test Precision : "
        f"{test_precision:.4f}"
    )

    print(
        f"Test Recall    : "
        f"{test_rec:.4f}"
    )

    print(
        f"Test F1        : "
        f"{test_f1:.4f}"
    )

    print(
        f"Test AUC       : "
        f"{test_auc:.4f}"
    )

    print(
        f"Final W_CNN    : "
        f"{model2.weight_cnn.item():.4f}"
    )

    print(
        f"Final W_TRANS  : "
        f"{model2.weight_transformer.item():.4f}"
    )


    # ========================================================
    # Save fold results
    # ========================================================

    fold_results.append(

        {

            "env":
            int(test_env),

            "test_acc":
            test_acc,

            "test_precision":
            test_precision,

            "test_rec":
            test_rec,

            "test_f1":
            test_f1,

            "test_auc":
            test_auc,

            "weight_cnn":
            float(
                model2
                .weight_cnn
                .item()
            ),

            "weight_transformer":
            float(
                model2
                .weight_transformer
                .item()
            )

        }

    )


    # ========================================================
    # Global ROC data
    # ========================================================

    all_test_labels_global.extend(

        test_trues_all
    )

    all_test_scores_global.extend(

        test_scores_all
    )


    # ========================================================
    # Save this fold's epoch history
    # ========================================================

    fold_history_df = pd.DataFrame(

        {

            "epoch":
            history_epoch,

            "train_loss":
            history_train_loss,

            "train_acc":
            history_train_acc,

            "train_recall":
            history_train_rec,

            "train_f1":
            history_train_f1,

            "test_acc":
            history_test_acc,

            "test_recall":
            history_test_rec,

            "test_f1":
            history_test_f1,

            "weight_cnn":
            history_weight_cnn,

            "weight_transformer":
            history_weight_transformer

        }

    )


    fold_history_df.to_excel(

        f"finetune_history_env_{test_env}.xlsx",

        index=False
    )


# ============================================================
# Summarize all LOSO folds
# ============================================================

results_df = pd.DataFrame(
    fold_results
)


print(
    "\n"
    "======================================================"
)

print(
    "各 env 最终测试结果"
)

print(
    "======================================================"
)


print(
    results_df
)


# ============================================================
# Mean performance
# ============================================================

print(
    "\n"
    "======================================================"
)

print(
    "平均测试性能"
)

print(
    "======================================================"
)


print(
    "Mean Acc:",
    results_df[
        "test_acc"
    ].mean()
)


print(
    "Mean Precision:",
    results_df[
        "test_precision"
    ].mean()
)


print(
    "Mean Recall:",
    results_df[
        "test_rec"
    ].mean()
)


print(
    "Mean F1:",
    results_df[
        "test_f1"
    ].mean()
)


print(
    "Mean AUC:",
    results_df[
        "test_auc"
    ].mean()
)


# ============================================================
# Save final LOSO results
# ============================================================

results_df.to_excel(

    "loso_finetune_test_results_no_leakage.xlsx",

    index=False
)


print(
    "\n"
    "LOSO 结果已保存："
    "loso_finetune_test_results_no_leakage.xlsx"
)


# ============================================================
# Overall ROC
# ============================================================

if (
    len(
        all_test_labels_global
    )
    >
    0
    and
    len(
        np.unique(
            all_test_labels_global
        )
    )
    >=
    2
):

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(

        all_test_labels_global,

        all_test_scores_global
    )


    auc_global = roc_auc_score(

        all_test_labels_global,

        all_test_scores_global
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(6, 6)
    )


    plt.plot(

        fpr,

        tpr,

        linewidth=2,

        label=(
            f"ROC curve "
            f"(AUC = "
            f"{auc_global:.4f})"
        )
    )


    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--",

        label="Random"
    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Overall ROC Curve "
        "(LOSO + Few-shot)"
    )


    plt.legend(
        loc="lower right"
    )


    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )


    plt.tight_layout()


    plt.savefig(

        "overall_ROC_no_test_label_leakage.png",

        dpi=300,

        bbox_inches="tight"
    )


    plt.show()


# ============================================================
# Finished
# ============================================================

print(
    "\n"
    "======================================================"
)

print(
    "LOSO + Few-shot 微调完成！"
)

print(
    "动态权重仅使用训练标签更新"
)

print(
    "测试标签只用于 Acc / Recall / F1 / AUC 计算"
)

print(
    "每个 epoch 均输出 Train/Test 指标"
)

print(
    "最终模型固定为第 "
    f"{num_epochs_finetune} "
    "个 epoch"
)

print(
    "======================================================"
)