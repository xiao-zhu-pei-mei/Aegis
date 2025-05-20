import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from matplotlib import font_manager as fm

# ========= 0. 全局可调 =========
ROOT_MODEL = "IFS_results"   # 保存最优模型根目录
DATA_BASE  = "data"   # 数据集目录

DATASETS = ["CKSAAP4", "CKSAAP5"]
METHODS  = ["shap", "anova", "relief"]
TITLE_MAP= {"shap": "SHAP", "anova": "ANOVA", "relief": "ReliefF"}

# ===== 字体 =====
REG_PATH  = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
BOLD_PATH = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"
if os.path.isfile(BOLD_PATH):
    fm.fontManager.addfont(BOLD_PATH)
    FONT_NAME = fm.FontProperties(fname=BOLD_PATH).get_name()
else:
    fm.fontManager.addfont(REG_PATH)
    FONT_NAME = fm.FontProperties(fname=REG_PATH).get_name()

plt.rcParams.update({"font.family": FONT_NAME, "font.weight": "bold"})
LINEWIDTH_AXIS = 2
FONTSIZE_TITLE = 45
FONTSIZE_TICK  = 45
CMAP = ListedColormap(['#0075ea', '#eb0077'])  # HC, PD
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========= 1. 与训练一致的 Transformer =========
class TransformerModel(nn.Module):
    """保持与训练阶段完全相同的参数命名，以便正确 load_state_dict"""

    def __init__(self, input_dim: int):
        super().__init__()
        nhead = max([h for h in range(8, 0, -1) if input_dim % h == 0] or [1])
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(self.transformer(x.unsqueeze(1)).squeeze(1))

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        return self.transformer(x.unsqueeze(1)).squeeze(1)

# ========= 2. 工具函数 =========

def load_best_n(dataset: str, method: str) -> int:
    info = os.path.join(ROOT_MODEL, dataset, method, "best_info.txt")
    with open(info) as fh:
        for line in fh:
            if line.startswith("best_n"):
                return int(line.strip().split("=")[1])
    raise RuntimeError(f"best_n not found in {info}")


def rank_file(dataset: str, method: str) -> str:
    # 排序文件命名样例 CKSAAP4_shap.csv / CKSAAP4_relief.csv / CKSAAP4_anova.csv
    return os.path.join(DATA_BASE, "train", f"{dataset}_{method}.csv")


def model_paths(dataset: str, method: str, n: int):
    mdir = os.path.join(ROOT_MODEL, dataset, method, "models")
    patt = re.compile(rf"top{n}_fold\d+\.pt")
    matched = sorted(p for p in os.listdir(mdir) if patt.match(p))
    return [os.path.join(mdir, m) for m in matched]

# ========= 3. 标签 =========
LABELS = {
    ds: pd.read_csv(os.path.join(DATA_BASE, "test", f"{ds}_with_header.csv"))["target"].values
    for ds in DATASETS
}

# ========= 4. 嵌入计算 =========
embeddings = {}
for ds in DATASETS:
    df_test = pd.read_csv(os.path.join(DATA_BASE, "test", f"{ds}_with_header.csv"))
    X_full = df_test.iloc[:, 1:].values
    feat2idx = {f: i for i, f in enumerate(df_test.columns[1:])}

    for method in METHODS:
        n = load_best_n(ds, method)
        rank_list = pd.read_csv(rank_file(ds, method))["feature"].tolist()[:n]
        idx_sel = [feat2idx[f] for f in rank_list if f in feat2idx]
        X_sel = X_full[:, idx_sel]

        reps_all = []
        for mp in model_paths(ds, method, n):
            model = TransformerModel(n)
            model.load_state_dict(torch.load(mp, map_location=DEVICE))
            model.to(DEVICE).eval()
            reps = model.encode(torch.tensor(X_sel, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            reps_all.append(reps)
            del model
            torch.cuda.empty_cache()

        reps_mean = np.mean(reps_all, axis=0)
        embeddings[(method, ds)] = umap.UMAP(random_state=42).fit_transform(reps_mean)

# ========= 5. 绘图 =========
fig, axes = plt.subplots(len(METHODS), len(DATASETS), figsize=(20, 25))
plt.subplots_adjust(left=0.04, right=0.84, top=0.97, bottom=0.03, wspace=0.25, hspace=0.25)

for i, method in enumerate(METHODS):
    for j, ds in enumerate(DATASETS):
        ax = axes[i, j]
        emb = embeddings[(method, ds)]
        ax.scatter(emb[:, 0], emb[:, 1], c=LABELS[ds], cmap=CMAP, s=60, alpha=0.85)
        ax.set_title(f"{TITLE_MAP[method]} - {ds}", fontsize=FONTSIZE_TITLE)
        ax.tick_params(axis='both', labelsize=FONTSIZE_TICK, width=LINEWIDTH_AXIS)
        for spine in ax.spines.values():
            spine.set_linewidth(LINEWIDTH_AXIS)

# 色条
for i in range(len(METHODS)):
    pos = axes[i, -1].get_position()
    cax = fig.add_axes([0.87, pos.y0, 0.015, pos.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0,1)), cax=cax, ticks=[0,1])
    cb.set_ticklabels(['HC', 'PD'])
    cb.ax.tick_params(labelsize=FONTSIZE_TICK-5)
    for t in cb.ax.get_yticklabels():
        t.set_fontweight('bold')

# ========= 保存 =========
os.makedirs("figure", exist_ok=True)
OUT = "figure/DL_UMAP_internal.svg"
plt.savefig(OUT, bbox_inches="tight")
plt.close()
print(f"✔ UMAP 保存至 {OUT}")
