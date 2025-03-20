import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import datetime


def plot_training_curves(df, output_dir="./figures"):
    """
    绘制符合学术规范的训练曲线图（损失、准确率、学习率）

    参数：
    df : DataFrame
        包含训练日志数据的Pandas DataFrame
    output_dir : str
        输出图片的保存目录（默认为当前目录下的figures文件夹）
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 设置学术论文绘图风格
    plt.style.use('seaborn-v0_8-paper')
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 12,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'savefig.format': 'pdf',  # 保存为矢量图
        'axes.linewidth': 1.5,
        'grid.linestyle': '--',
        'grid.alpha': 0.6
    })

    # 定义列名（根据实际数据修改）
    epoch_col = 'epoch'
    train_loss_col = 'train_loss'
    val_loss_col = 'val_loss'
    val_acc_col = 'val_accuracy'
    lr_col = 'lr'

    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 绘制损失曲线
    plt.figure()
    plt.plot(df[epoch_col], df[train_loss_col],
             color='#E64B35', linewidth=2, label='Training Loss')
    plt.plot(df[epoch_col], df[val_loss_col],
             color='#4DBBD5', linewidth=2, linestyle='--', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(frameon=True, loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"loss_curve_{timestamp}.pdf")
    plt.close()

    # 绘制准确率曲线
    plt.figure()
    plt.plot(df[epoch_col], df[val_acc_col],
             color='#00A087', linewidth=2, marker='o', markersize=6,
             markevery=5, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)  # 准确率范围标准化
    plt.legend(frameon=True, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"accuracy_curve_{timestamp}.pdf")
    plt.close()

    # 绘制学习率曲线（对数坐标）
    plt.figure()
    plt.semilogy(df[epoch_col], df[lr_col],
                 color='#7E6148', linewidth=2, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend(frameon=True, loc='upper right')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"lr_curve_{timestamp}.pdf")
    plt.close()


if __name__ == "__main__":

    df = pd.read_csv('training_results/training_log_20250315_161016.csv')
    plot_training_curves(df, output_dir="visualization_results/figures")