import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.metrics import roc_auc_score

# ----------------------- 可调参数 -----------------------
ROOT_DIR   = "IFS_results"      # 预测结果根目录
TEST_DIR   = "data/test"   # *_with_header.csv 标签文件
OUTPUT_SVG = "figure/IFS_curves.svg"

DATASETS = ["CKSAAP4", "CKSAAP5"]
METHODS  = ["shap", "anova", "relief"]
METHOD_LABELS = {"shap": "SHAP", "relief": "ReliefF", "anova": "ANOVA"}

# ======== 字体设置：Times New Roman Bold ========
# Ubuntu 下 msttcorefonts 包通常只含 Regular/Italic；
# 若系统安装了粗体版本，可指向其绝对路径；否则让 matplotlib 强制加粗渲染。
REG_PATH  = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
BOLD_PATH = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"  # 若无则回退

if os.path.isfile(BOLD_PATH):
    fm.fontManager.addfont(BOLD_PATH)
    tnr_name = fm.FontProperties(fname=BOLD_PATH).get_name()
else:
    if os.path.isfile(REG_PATH):
        fm.fontManager.addfont(REG_PATH)
        tnr_name = fm.FontProperties(fname=REG_PATH).get_name()
        print("⚠️  Bold TNR 未找到，使用 Regular 并强制加粗渲染。")
    else:
        tnr_name = "Times New Roman"  # 由系统字体匹配
        print("⚠️  Times New Roman 字体文件未找到，依赖系统字体。")

plt.rcParams.update({
    "font.family":       tnr_name,
    "font.weight":       "bold",    # 全局加粗
    "axes.labelweight":  "bold",
    "axes.titleweight":  "bold",
})

# ======== 颜色 & 线型 ======== & 线型 ========
COLORS      = ["#CC3366", "#FFCC00", "#009999"]  # shap, anova, relief 对应顺序
LINESTYLES  = ["-", "-", "-"]
FONTSIZE    = 34

# 手动注释位置：1~8
MANUAL_POS = {
    "CKSAAP4": {"anova": 6, "relief": 5, "shap": 1},
    "CKSAAP5": {"anova": 6, "relief": 1, "shap": 5},
}

# 偏移映射
OFFSET_MAP = {
    1: (  5,  5, "left",   "bottom"),
    2: ( -5,  5, "right",  "bottom"),
    3: ( -5, -5, "right",  "top"   ),
    4: (  5, -5, "left",   "top"   ),
    5: (  0,  5, "center", "bottom"),
    6: (  0, -5, "center", "top"   ),
    7: (  2,  8, "left",   "top"   ),
    8: (  5, 10, "left",   "top"   ),
}

# =============== 工具函数 ===============

def load_test_labels(dataset: str) -> np.ndarray:
    """读取测试集标签"""
    path = os.path.join(TEST_DIR, f"{dataset}_with_header.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)["target"].values


def gather_auc_table() -> pd.DataFrame:
    """遍历 ROOT_DIR 生成 AUC 表"""
    records = []
    for ds in DATASETS:
        y_true = load_test_labels(ds)
        for method in METHODS:
            m_dir = os.path.join(ROOT_DIR, ds, method)
            if not os.path.isdir(m_dir):
                continue
            for fname in os.listdir(m_dir):
                m = re.match(r"top(\d+)_pred\.csv", fname)
                if not m:
                    continue
                n_feat = int(m.group(1))
                df_pred = pd.read_csv(os.path.join(m_dir, fname))
                y_score = df_pred["Score_1"].values
                auc = roc_auc_score(y_true, y_score)
                records.append((ds, method, n_feat, auc))
    return pd.DataFrame(records, columns=["Dataset", "Method", "N_Features", "AUC"])

# =============== 主绘图函数 ===============

def plot_ifs_curves():
    os.makedirs(os.path.dirname(OUTPUT_SVG), exist_ok=True)
    df = gather_auc_table()
    if df.empty:
        raise RuntimeError("No predictions found, please run inference first.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(wspace=0.18)

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        sub_all = df[df["Dataset"] == dataset]
        if sub_all.empty:
            ax.set_title(f"{dataset}\n[No Data]", fontsize=FONTSIZE)
            ax.set_xlabel("Number of Features", fontsize=FONTSIZE, fontstyle="italic")
            ax.set_ylabel("AUC", fontsize=FONTSIZE, fontstyle="italic")
            continue

        max_feat = sub_all["N_Features"].max()

        for m_idx, method in enumerate(METHODS):
            sub = sub_all[sub_all["Method"] == method].copy()
            if sub.empty:
                continue
            sub.sort_values("N_Features", inplace=True)
            x = sub["N_Features"].to_numpy()
            y = sub["AUC"].to_numpy()

            x_plot = np.insert(x, 0, 0)  # 曲线从 (0,0)
            y_plot = np.insert(y, 0, 0)

            ax.plot(x_plot, y_plot,
                    label=METHOD_LABELS[method],
                    color=COLORS[m_idx],
                    linestyle=LINESTYLES[m_idx],
                    linewidth=2)

            best_i = y.argmax()
            bx, by = x[best_i], y[best_i]
            ax.scatter(bx, by, color="green", alpha=0.6)
            ax.hlines(by, -40, bx, colors="gray", linestyles="--")
            ax.vlines(bx, 0, by, colors="gray", linestyles="--")

            pos_code = MANUAL_POS.get(dataset, {}).get(method, 1)
            dx, dy, ha, va = OFFSET_MAP.get(pos_code, (5, 5, "left", "bottom"))
            ax.annotate(f"({bx}, {by:.3f})", xy=(bx, by),
                        xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=30, fontweight="bold")

        ax.set_xlim(-40, max_feat + 5)
        ax.set_ylim(0.65, 1.02)
        ax.set_title(dataset, fontsize=FONTSIZE)
        ax.set_xlabel("Number of Features", fontsize=FONTSIZE, fontstyle="italic")
        ax.set_ylabel("AUC", fontsize=FONTSIZE, fontstyle="italic")
        ax.tick_params(axis="both", labelsize=FONTSIZE)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        legend = ax.legend(fontsize=FONTSIZE, loc="lower right", frameon=False)
        for txt in legend.get_texts():
            txt.set_fontweight("bold")
        for line in legend.get_lines():
            line.set_linewidth(4)

    plt.tight_layout()
    plt.savefig(OUTPUT_SVG)
    plt.close()
    print(f"✅ IFS curves saved to {OUTPUT_SVG}")


if __name__ == "__main__":
    plot_ifs_curves()
