import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score
)

from IRM import irm_loss

from functions2 import (
    self_sliding,
    CNN,
    Transformer,
    dynamic_weight_adjustment,
    compute_accuracy
)


# ======================================================
# Basic settings
# ======================================================

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

batch_size = 128
num_epoch = 65

# IRM penalty weight
lambda_irm = 0.05

# Initial learning rate
initial_lr = 2e-3


# ======================================================
# Load public dataset
# ======================================================

data = pd.read_excel(
    'Upfall4完整版数据(标注01）.xlsx'
)


# ======================================================
# Read features / labels / environments
# ======================================================

# IMU features
X = data.iloc[:, 1:13].values

# Label
y = data.iloc[:, 13].values

# User ID as environment
env_labels = data.iloc[:, 14].values


print(
    "Users:",
    np.unique(env_labels)
)


# ======================================================
# Normalize
# ======================================================

scaler = StandardScaler()

X = scaler.fit_transform(X)


# ======================================================
# Sliding window
# ======================================================

X_sliding = self_sliding(
    X,
    window_width=200,
    stride=100
)

y_sliding = self_sliding(
    y,
    window_width=200,
    stride=100
)

env_sliding = self_sliding(
    env_labels,
    window_width=200,
    stride=100
)


# ======================================================
# Get sliding-window dimensions
# ======================================================

num_samples = X_sliding.shape[0]

num_windows = X_sliding.shape[1]


# ======================================================
# Reshape
# ======================================================

X_all = np.reshape(
    X_sliding,
    (
        num_samples * num_windows,
        12
    )
)

y_all = np.reshape(
    y_sliding,
    (-1,)
)

env_all = np.reshape(
    env_sliding,
    (-1,)
)


# Ensure correct data types
X_all = X_all.astype(
    np.float32
)

y_all = y_all.astype(
    np.int64
)

env_all = env_all.astype(
    np.int64
)


print(
    "Total samples:",
    X_all.shape[0]
)

print(
    "X shape:",
    X_all.shape
)

print(
    "y shape:",
    y_all.shape
)

print(
    "env shape:",
    env_all.shape
)


# ======================================================
# Dataset
# ======================================================

dataset = torch.utils.data.TensorDataset(

    torch.tensor(
        X_all,
        dtype=torch.float32
    ),

    torch.tensor(
        y_all,
        dtype=torch.long
    ),

    torch.tensor(
        env_all,
        dtype=torch.long
    )
)


loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# ======================================================
# Model
# CNN + Transformer + Dynamic Weight Fusion
# ======================================================

class MyNet(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim
    ):

        super().__init__()

        # --------------------------------------------------
        # CNN branch
        # --------------------------------------------------

        self.cnn = CNN(
            input_dim,
            output_dim
        )

        # --------------------------------------------------
        # Transformer branch
        # --------------------------------------------------

        self.transformer = Transformer(
            input_dim,
            output_dim,
            nhead=12
        )

        # --------------------------------------------------
        # Dynamic fusion weights
        #
        # These are nn.Parameter, so they will be saved into
        # model.state_dict():
        #
        # weight_cnn
        # weight_transformer
        # --------------------------------------------------

        self.weight_cnn = nn.Parameter(
            torch.ones(1)
        )

        self.weight_transformer = nn.Parameter(
            torch.ones(1)
        )

        # --------------------------------------------------
        # Dropout
        # --------------------------------------------------

        self.dropout = nn.Dropout(
            p=0.3
        )

        # --------------------------------------------------
        # Final classifier
        # --------------------------------------------------

        self.classifier = nn.Linear(
            output_dim,
            output_dim
        )


    def forward(
        self,
        x,
        ground_truth_labels
    ):

        # ==================================================
        # CNN branch
        # ==================================================

        cnn_output = self.cnn(
            x
        )


        # ==================================================
        # Transformer branch
        # ==================================================

        transformer_output = self.transformer(
            x
        )


        # ==================================================
        # Calculate current branch accuracy
        # ==================================================

        cnn_accuracy = compute_accuracy(
            cnn_output,
            ground_truth_labels
        )

        transformer_accuracy = compute_accuracy(
            transformer_output,
            ground_truth_labels
        )


        # ==================================================
        # Dynamic weight adjustment
        # ==================================================
        #
        # Keep the same dynamic-weight mechanism used
        # by your fine-tuning code.
        # ==================================================

        self.weight_cnn.data = dynamic_weight_adjustment(
            self.weight_cnn,
            cnn_accuracy
        )

        self.weight_transformer.data = dynamic_weight_adjustment(
            self.weight_transformer,
            transformer_accuracy
        )


        # ==================================================
        # Dynamic fusion
        # ==================================================

        feature = (
            self.weight_cnn * cnn_output
            +
            self.weight_transformer * transformer_output
        )


        # ==================================================
        # Dropout
        # ==================================================

        feature = self.dropout(
            feature
        )


        # ==================================================
        # Classification
        # ==================================================

        output = self.classifier(
            feature
        )


        return output


# ======================================================
# Initialize model
# ======================================================

model = MyNet(
    input_dim=12,
    output_dim=2
).to(device)


print("\n===== Model =====")

print(model)


# ======================================================
# Class imbalance weight
# ======================================================

class_count = np.bincount(
    y_all.astype(int)
)


# Prevent division by zero
if len(class_count) < 2:

    raise ValueError(
        "Training data must contain both class 0 and class 1."
    )


if np.any(class_count == 0):

    raise ValueError(
        f"Some class has zero samples: {class_count}"
    )


class_weight = (
    len(y_all)
    /
    (2 * class_count)
)


class_weight = torch.tensor(
    class_weight,
    dtype=torch.float32
).to(device)


print(
    "Class count:",
    class_count
)

print(
    "Class weight:",
    class_weight.cpu().numpy()
)


# ======================================================
# Loss
# ======================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weight
)


# ======================================================
# Optimizer
# ======================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=initial_lr,
    weight_decay=1e-4
)


# ======================================================
# Learning rate scheduler
# ======================================================
#
# Epoch 1-20  : 2e-3
# Epoch 21-40 : 1e-3
# Epoch 41-60 : 5e-4
# Epoch 61-65 : 2.5e-4
#
# ======================================================

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.5
)


# ======================================================
# Training
# ======================================================

best_loss = float("inf")


history = {

    "epoch": [],

    "loss": [],

    "ce_loss": [],

    "irm_loss": [],

    "lr": [],

    "acc": [],

    "recall": [],

    "f1": [],

    # Save dynamic weights for observation
    "weight_cnn": [],

    "weight_transformer": []
}


for epoch in range(num_epoch):

    # ==================================================
    # Training mode
    # ==================================================

    model.train()


    epoch_loss = []

    epoch_ce_loss = []

    epoch_irm_loss = []

    preds = []

    labels_record = []


    # --------------------------------------------------
    # Current learning rate
    # --------------------------------------------------

    current_lr = optimizer.param_groups[0]["lr"]


    # ==================================================
    # Mini-batch training
    # ==================================================

    for inputs, labels, groups in loader:

        # --------------------------------------------------
        # Move data to device
        # --------------------------------------------------

        inputs = inputs.to(
            device
        )

        labels = labels.to(
            device
        )

        groups = groups.to(
            device
        )


        # --------------------------------------------------
        # Clear gradients
        # --------------------------------------------------

        optimizer.zero_grad()


        # --------------------------------------------------
        # Forward
        #
        # IMPORTANT:
        # labels must be passed into MyNet because
        # dynamic fusion needs branch accuracy.
        # --------------------------------------------------

        outputs = model(
            inputs,
            labels
        )


        # --------------------------------------------------
        # Cross-entropy loss
        # --------------------------------------------------

        ce = criterion(
            outputs,
            labels
        )


        # --------------------------------------------------
        # IRM penalty
        # --------------------------------------------------

        irm = irm_loss(
            outputs,
            labels,
            groups
        )


        # --------------------------------------------------
        # Total loss
        # --------------------------------------------------

        loss = (
            ce
            +
            lambda_irm * irm
        )


        # --------------------------------------------------
        # Backward
        # --------------------------------------------------

        loss.backward()


        # --------------------------------------------------
        # Update parameters
        # --------------------------------------------------

        optimizer.step()


        # --------------------------------------------------
        # Record losses
        # --------------------------------------------------

        epoch_loss.append(
            loss.item()
        )

        epoch_ce_loss.append(
            ce.item()
        )

        epoch_irm_loss.append(
            irm.item()
        )


        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        pred = torch.argmax(
            outputs,
            dim=1
        )


        preds.extend(
            pred.detach().cpu().numpy()
        )

        labels_record.extend(
            labels.detach().cpu().numpy()
        )


    # ==================================================
    # Training metrics
    # ==================================================

    avg_loss = np.mean(
        epoch_loss
    )

    avg_ce = np.mean(
        epoch_ce_loss
    )

    avg_irm = np.mean(
        epoch_irm_loss
    )


    acc = accuracy_score(
        labels_record,
        preds
    )


    recall = recall_score(
        labels_record,
        preds,
        zero_division=0
    )


    f1 = f1_score(
        labels_record,
        preds,
        zero_division=0
    )


    # ==================================================
    # Get current dynamic weights
    # ==================================================

    current_weight_cnn = model.weight_cnn.detach().cpu().item()

    current_weight_transformer = (
        model.weight_transformer
        .detach()
        .cpu()
        .item()
    )


    # ==================================================
    # Print current epoch
    # ==================================================

    print(
        f"Epoch {epoch + 1:02d}/{num_epoch} | "
        f"LR:{current_lr:.6f} | "
        f"Loss:{avg_loss:.4f} | "
        f"CE:{avg_ce:.4f} | "
        f"IRM:{avg_irm:.4f} | "
        f"Acc:{acc:.4f} | "
        f"Recall:{recall:.4f} | "
        f"F1:{f1:.4f} | "
        f"W_CNN:{current_weight_cnn:.4f} | "
        f"W_TRANS:{current_weight_transformer:.4f}"
    )


    # ==================================================
    # Save history
    # ==================================================

    history["epoch"].append(
        epoch + 1
    )

    history["loss"].append(
        avg_loss
    )

    history["ce_loss"].append(
        avg_ce
    )

    history["irm_loss"].append(
        avg_irm
    )

    history["lr"].append(
        current_lr
    )

    history["acc"].append(
        acc
    )

    history["recall"].append(
        recall
    )

    history["f1"].append(
        f1
    )

    history["weight_cnn"].append(
        current_weight_cnn
    )

    history["weight_transformer"].append(
        current_weight_transformer
    )


    # ==================================================
    # Save best pretrained model
    # ==================================================

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "IRM_pretrained_dynamic.pth"
        )

        print(
            f"  >>> Best model updated, "
            f"loss = {best_loss:.6f}"
        )


    # ==================================================
    # Update learning rate
    # ==================================================

    scheduler.step()


# ======================================================
# Save training history
# ======================================================

df = pd.DataFrame(
    history
)

df.to_excel(
    "IRM_pretraining_record.xlsx",
    index=False
)


# ======================================================
# Plot training loss curve
# ======================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["epoch"],
    history["loss"],
    linewidth=2,
    label="Total Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("IRM Pretraining Loss Curve")

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "IRM_pretraining_loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ======================================================
# Verify saved pretrained model
# ======================================================

print(
    "\n================================="
)

print(
    "Checking saved state_dict..."
)

saved_state_dict = torch.load(
    "IRM_pretrained_dynamic.pth",
    map_location="cpu"
)

if "weight_cnn" not in saved_state_dict:
    raise RuntimeError(
        "weight_cnn was not saved in IRM_pretrained.pth"
    )

if "weight_transformer" not in saved_state_dict:
    raise RuntimeError(
        "weight_transformer was not saved in IRM_pretrained.pth"
    )

print(
    "Saved weight_cnn:",
    saved_state_dict["weight_cnn"]
)

print(
    "Saved weight_transformer:",
    saved_state_dict["weight_transformer"]
)


# ======================================================
# Test strict loading
# ======================================================

check_model = MyNet(
    input_dim=12,
    output_dim=2
).to("cpu")

check_model.load_state_dict(
    saved_state_dict,
    strict=True
)

print(
    "Checkpoint strict loading test: PASSED"
)


# ======================================================
# Finished
# ======================================================

print(
    "================================="
)

print(
    "IRM pretraining finished!"
)

print(
    f"Best training loss: {best_loss:.6f}"
)

print(
    "Saved model: IRM_pretrained.pth"
)

print(
    "Saved record: IRM_pretraining_record.xlsx"
)

print(
    "Saved loss curve: IRM_pretraining_loss_curve.png"
)