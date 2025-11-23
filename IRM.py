import torch
import torch.nn as nn
import torch.nn.functional as F

# 在计算完传统loss后 引入环境标签，相当于计算两次损失
import torch.nn.functional as F

def irm_loss(outputs, labels, groups):
    loss = nn.CrossEntropyLoss()

    class_weights = torch.tensor([1.0, 3.0])

    # Apply class weights to the loss function
    weighted_loss = nn.CrossEntropyLoss(weight=class_weights)  # 将类别权重应用与损失权重

    task_loss = weighted_loss(outputs, labels.long())

    group_ids = torch.unique(groups)
    num_groups = len(group_ids)

    group_losses = []
    for group_id in group_ids:
        group_mask = (groups == group_id)
        group_outputs = outputs[group_mask]
        group_labels = labels[group_mask]
        group_loss = weighted_loss(group_outputs, group_labels.long())
        group_losses.append(group_loss)

    violation = torch.sum(torch.stack(group_losses)) / num_groups

    irm_lambda = 0.1  # Adjust the weight of the IRM loss
    total_loss = task_loss + irm_lambda * violation

    return total_loss


