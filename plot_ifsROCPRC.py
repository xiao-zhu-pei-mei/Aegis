import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

# ========= 全局字体 =========
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

# ========= 颜色、线型与字号 =========
COLORS = {"shap": "#CC3366", "anova": "#FFCC00", "relief": "#009999"}
LINEWIDTH = 4
FS_AXIS   = 48
FS_TITLE  = 48
FS_LEGEND = 36

# ========= 路径与数据集 =========
ROOT_DIR  = "IFS_results"          # 预测结果根目录 (CKSAAP*/method/topN_pred.csv)
TEST_DIR  = "data/test"       # 包含 *with_header.csv 的标签
DATASETS  = ["CKSAAP4", "CKSAAP5"]
METHODS   = ["shap", "anova", "relief"]
LABEL_COL = "target"
PROB_COL  = "Score_1"

# ========= 搜索最佳模型并计算曲线 =========

def load_labels(dataset):
    path = os.path.join(TEST_DIR, f"{dataset}_with_header.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)[LABEL_COL].values


def best_prediction_file(dataset, method):
    """返回给定数据集+方法 AUC 最高的 prediction csv 完整路径及其 AUC"""
    method_dir = os.path.join(ROOT_DIR, dataset, method)
    if not os.path.isdir(method_dir):
        raise FileNotFoundError(method_dir)
    label = load_labels(dataset)
    best_auc, best_path = -1.0, None
    for fname in os.listdir(method_dir):
        if not re.match(r"top\d+_pred\.csv", fname):
            continue
        fpath = os.path.join(method_dir, fname)
        score = pd.read_csv(fpath)[PROB_COL].values
        auc_val = auc(*roc_curve(label, score)[:2])
        if auc_val > best_auc:
            best_auc, best_path = auc_val, fpath
    if best_path is None:
        raise RuntimeError(f"No prediction files found in {method_dir}")
    return best_path, best_auc


def compute_curves(label, score):
    fpr, tpr, _ = roc_curve(label, score)
    roc_auc = auc(fpr, tpr)
    rec, prec, _ = precision_recall_curve(label, score)
    ap = average_precision_score(label, score)
    return {
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "rec": rec, "prec": prec, "ap": ap,
    }

# gather all curves
curves = {}
for dataset in DATASETS:
    label = load_labels(dataset)
    curves[dataset] = {}
    for method in METHODS:
        pred_path, best_auc = best_prediction_file(dataset, method)
        score = pd.read_csv(pred_path)[PROB_COL].values
        curves[dataset][method] = compute_curves(label, score)
        print(f"{dataset}-{method}: best AUC={best_auc:.4f}  -> {os.path.basename(pred_path)}")

# ========= 绘图 =========
fig, axes = plt.subplots(2, 2, figsize=(20, 18))
plt.subplots_adjust(hspace=0.5, wspace=1)
axes = axes.flatten()

subplot_cfg = {
    0: ("CKSAAP4", "roc"), 1: ("CKSAAP4", "prc"),
    2: ("CKSAAP5", "roc"), 3: ("CKSAAP5", "prc"),
}

for idx, ax in enumerate(axes):
    dataset, ptype = subplot_cfg[idx]
    for method in METHODS:
        cdict = curves[dataset][method]
        color = COLORS[method]
        label_name = "ReliefF" if method == "relief" else method.upper()
        if ptype == "roc":
            ax.plot(cdict["fpr"], cdict["tpr"], color=color, linewidth=LINEWIDTH,
                    label=f"{label_name} (AUC = {cdict['roc_auc']:.3f})")
            ax.set_xlabel("False Positive Rate", fontsize=FS_AXIS, fontstyle="italic")
            ax.set_ylabel("True Positive Rate", fontsize=FS_AXIS, fontstyle="italic")
        else:
            ax.step(cdict["rec"], cdict["prec"], where="post", color=color, linewidth=LINEWIDTH,
                    label=f"{label_name} (AP = {cdict['ap']:.3f})")
            ax.set_xlabel("Recall", fontsize=FS_AXIS, fontstyle="italic")
            ax.set_ylabel("Precision", fontsize=FS_AXIS, fontstyle="italic")
            ax.set_ylim([-0.05, 1.05])

    ax.set_title(f"{dataset} - {'ROC' if ptype=='roc' else 'PRC'}",
                 fontsize=FS_TITLE)
    ax.tick_params(axis="both", labelsize=FS_AXIS, width=2)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    legend = ax.legend(fontsize=FS_LEGEND, loc="lower center", frameon=False)
    for line in legend.get_lines():
        line.set_linewidth(LINEWIDTH * 2)
        line.set_alpha(0.9)

# ========= 保存 =========
os.makedirs("figure", exist_ok=True)
OUT_SVG = "figure/CKSAAP_ROC_PRC.svg"
plt.tight_layout()
plt.savefig(OUT_SVG, bbox_inches="tight")
plt.close()
print(f"✔ 已保存 {OUT_SVG}")
