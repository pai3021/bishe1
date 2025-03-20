import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import time

# 自定义模块
from models.resnet1d import ResNet1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def save_training_log(epoch, train_loss, val_loss, val_report, filename):
    """保存训练日志到CSV文件（新增val_loss记录）"""
    log_entry = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_report['accuracy'],
        'val_f1_macro': val_report['macro avg']['f1-score'],
        'val_precision_macro': val_report['macro avg']['precision'],
        'val_recall_macro': val_report['macro avg']['recall'],
        'lr': optimizer.param_groups[0]['lr']  # 记录当前学习率
    }

    if not os.path.isfile(filename):
        pd.DataFrame([log_entry]).to_csv(filename, index=False)
    else:
        pd.DataFrame([log_entry]).to_csv(filename, mode='a', header=False, index=False)


def evaluate_model(model, dataloader, device, criterion=None):
    """
    增强版评估函数
    返回：
    - 如果提供criterion：返回(report, cm, loss)
    - 否则：返回(report, cm)
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    if criterion is not None:
        avg_loss = total_loss / len(dataloader.dataset)
        return report, cm, avg_loss
    else:
        return report, cm


if __name__ == "__main__":
    # 超参数设置
    batch_size = 32
    n_epoch = 50
    initial_lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建保存目录
    save_dir = "training_results"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # # 数据加载部分（示例数据）
    # X_train = np.random.randn(1000, 1, 5000)
    # y_train = np.random.randint(0, 5, 1000)
    # X_val = np.random.randn(200, 1, 5000)
    # y_val = np.random.randint(0, 5, 200)
    # 加载数据
    print("---loading data---")
    X_train = np.load('E:/deep-learning/tsai-main/data/X_train.npy')
    y_train = np.load('E:/deep-learning/tsai-main/data/y_train.npy')
    X_val = np.load('E:/deep-learning/tsai-main/data/X_val.npy')
    y_val = np.load('E:/deep-learning/tsai-main/data/y_val.npy')

    # 创建数据集
    train_dataset = MyDataset(X_train, y_train)
    val_dataset = MyDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 模型初始化
    model = ResNet1D(
        in_channels=1,
        base_filters=128,
        kernel_size=16,
        stride=2,
        groups=32,
        n_block=48,
        n_classes=5,
        downsample_gap=6,
        increasefilter_gap=12,
        use_do=True
    ).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-3)
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',    # 根据验证集F1分数调整（max模式）
        factor=0.5,   # 学习率衰减因子
        patience=5,   # 等待5个epoch无改善
        verbose=True  # 显示调整信息
    )

    best_val_f1 = 0.0
    log_file = os.path.join(save_dir, f"training_log_{timestamp}.csv")

    # 训练循环（含学习率调整）===============================================
    for epoch in range(n_epoch):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epoch}")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": optimizer.param_groups[0]['lr']
            })

        train_loss = epoch_loss / len(train_dataset)

        # 验证阶段（同时计算损失）
        val_report, val_cm, val_loss = evaluate_model(
            model, val_loader, device, criterion
        )

        # 更新学习率（根据验证集F1分数）=====================================
        current_f1 = val_report['macro avg']['f1-score']
        scheduler.step(current_f1)  # 注意这里使用F1分数作为调度依据

        # 保存日志（新增val_loss）
        save_training_log(epoch+1, train_loss, val_loss, val_report, log_file)

        # 打印验证结果
        print(f"\nValidation Results - Epoch {epoch+1}")
        print(f"Loss: {val_loss:.4f} | Acc: {val_report['accuracy']:.4f} | F1: {current_f1:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print("Confusion Matrix:")
        print(val_cm)

        # 保存最佳模型
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            model_save_path = os.path.join(save_dir, f"best_model_{timestamp}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"?? New best model saved (F1: {best_val_f1:.4f})")

    # 最终结果保存
    print("\nTraining Completed!")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Final Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")