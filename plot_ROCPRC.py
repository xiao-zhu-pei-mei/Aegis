import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from matplotlib import font_manager as fm

# ========= 字体设置：Times New Roman Bold =========
REG_PATH  = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
BOLD_PATH = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"
if os.path.isfile(BOLD_PATH):
    fm.fontManager.addfont(BOLD_PATH)
    FONT_NAME = fm.FontProperties(fname=BOLD_PATH).get_name()
else:
    fm.fontManager.addfont(REG_PATH)
    FONT_NAME = fm.FontProperties(fname=REG_PATH).get_name()
    print("⚠️  Bold TNR 未找到，使用 Regular 并加粗渲染。")

plt.rcParams.update({
    "font.family":      FONT_NAME,
    "font.weight":      "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

# ========= 变量 =========
ml_models = ["XGBoost", "RF", "LightGBM", "SVM"]
dl_models = ["CNN", "ABCNN", "RNN", "BRNN", "AE", "Transformer"]
features   = ["CKSAAP4", "CKSAAP5", "CTDC", "CTDD", "CTDT", "PAAC"]
colors     = ['#FF6600', '#33CC00', '#9900CC', '#FF0000', '#0066FF', '#00CCCC']

# 读取标签
y_true = pd.read_csv('data/test/CKSAAP4.csv', header=None)[0].values

# ========= 工具函数 =========

def get_roc_pr_data(model: str, feature: str, y_true: np.ndarray, base_dir: str = "probs"):
    filepath = os.path.join(base_dir, f"{model}_{feature}")
    df_pred = pd.read_csv(filepath, sep='\t')
    y_score = df_pred["Score_1"].values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return fpr, tpr, roc_auc, precision, recall, ap


def plot_roc_pr_for_model(ax, model: str, y_true: np.ndarray, features: list[str], plot_type: str, base_dir="probs"):
    fontsize = 30
    legend_fontsize = 22
    for idx, feature in enumerate(features):
        fpr, tpr, roc_auc, precision, recall, ap = get_roc_pr_data(model, feature, y_true, base_dir)
        if plot_type == 'roc':
            ax.plot(fpr, tpr, label=f"{feature} (AUC={roc_auc:.3f})", lw=3, color=colors[idx])
        else:  # prc
            ax.plot(recall, precision, label=f"{feature} (AP={ap:.3f})", lw=3, color=colors[idx])

    if plot_type == 'roc':
        ax.set_xlabel('False Positive Rate', fontsize=fontsize, fontstyle='italic')
        ax.set_ylabel('True Positive Rate',  fontsize=fontsize, fontstyle='italic')
        ax.set_title(f'{model} - ROC', fontsize=fontsize)
    else:
        ax.set_xlabel('Recall',    fontsize=fontsize, fontstyle='italic')
        ax.set_ylabel('Precision', fontsize=fontsize, fontstyle='italic')
        ax.set_title(f'{model} - PRC', fontsize=fontsize)
        ax.set_ylim([0.0, 1.05])

    ax.tick_params(axis='both', labelsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    legend = ax.legend(fontsize=legend_fontsize, loc='lower center', frameon=False)
    for line in legend.get_lines():
        line.set_linewidth(8)

# ========= 主绘图 =========

def ensure_fig_dir():
    os.makedirs("figure", exist_ok=True)


def plot_all():
    ensure_fig_dir()

    # 机器学习模型 (2 行 × 4 列)
    fig_ml, axes_ml = plt.subplots(2, 4, figsize=(26, 12))
    axes_ml = axes_ml.reshape(2, 4)
    for i, model in enumerate(ml_models):
        r, c = divmod(i, 2)
        plot_roc_pr_for_model(axes_ml[r, c*2],     model, y_true, features, 'roc')
        plot_roc_pr_for_model(axes_ml[r, c*2 + 1], model, y_true, features, 'prc')
    plt.tight_layout()
    fig_ml.savefig("figure/ML_ROCPRC.svg", bbox_inches='tight')
    plt.close(fig_ml)

    # 深度学习模型 (3 行 × 4 列)
    fig_dl, axes_dl = plt.subplots(3, 4, figsize=(26, 18))
    axes_dl = axes_dl.reshape(3, 4)
    for i, model in enumerate(dl_models):
        r, c = divmod(i, 2)
        plot_roc_pr_for_model(axes_dl[r, c*2],     model, y_true, features, 'roc')
        plot_roc_pr_for_model(axes_dl[r, c*2 + 1], model, y_true, features, 'prc')
    plt.tight_layout()
    fig_dl.savefig("figure/DL_ROCPRC.svg", bbox_inches='tight')
    plt.close(fig_dl)
    print("✔ ROC/PRC 图已保存至 figure/*.svg")


if __name__ == "__main__":
    plot_all()
