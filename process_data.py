import os
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

# ========================= 0. 全局路径 & 参数 =========================
RAW_DIR   = Path("data/raw")        # 原始 fasta 路径
OUT_ROOT  = Path("data")             # 特征与排序输出根目录
TRAIN_FASTA = RAW_DIR / "training.fasta"  # 原始训练 fasta
TEST_FASTA  = RAW_DIR / "testing.fasta"   # 原始测试 fasta

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAIRS = ["".join(p) for p in itertools.product(AMINO_ACIDS, repeat=2)]  # 400 种

K_SPACES = [4, 5]  # k=4 -> CKSAAP4, k=5 -> CKSAAP5

LABEL_MAP = {"POS": 1, "OTH": 1, "NEG": 0}

# ========================= 1. FASTA 解析 =========================

def parse_fasta(path: Path):
    """解析 FASTA，返回 [(id, label, seq)]"""
    records = []
    with open(path) as f:
        current_id, current_lab = None, None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                parts = line[1:].split("|")
                tag3 = parts[0][:3]
                if tag3 not in LABEL_MAP:
                    current_id = None
                    continue
                current_id = parts[0]
                current_lab = LABEL_MAP[tag3]
            else:
                if current_id is not None:
                    records.append((current_id, current_lab, line.upper()))
    return records

# ========================= 2. CKSAAP 特征提取 =========================

def ck_saap_vector(seq: str, k: int):
    """给定序列和 k，返回长度 400*(k+1) 的向量 (频率)"""
    vec = np.zeros(400 * (k + 1), dtype=float)
    L = len(seq)
    for kk in range(k + 1):
        if L <= kk + 1:
            continue
        sub_len = L - kk - 1
        offset = kk * 400
        counts = dict.fromkeys(PAIRS, 0)
        for i in range(sub_len):
            pair = seq[i] + seq[i + kk + 1]
            if pair in counts:
                counts[pair] += 1
        # 频率
        for p_idx, pair in enumerate(PAIRS):
            vec[offset + p_idx] = counts[pair] / sub_len
    return vec


def build_feature_df(records, k: int):
    rows = []
    for _id, lab, seq in records:
        feat = ck_saap_vector(seq, k)
        rows.append(np.concatenate([[lab], feat]))
    cols = ["target"] + [f"{pair}_{kk}" for kk in range(k + 1) for pair in PAIRS]
    return pd.DataFrame(rows, columns=cols)

# ========================= 3. 生成 CKSAAP CSV =========================

def prepare_cksaap_files():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "train").mkdir(exist_ok=True)
    (OUT_ROOT / "test").mkdir(exist_ok=True)

    train_records = parse_fasta(TRAIN_FASTA)
    test_records  = parse_fasta(TEST_FASTA)

    for k in K_SPACES:
        print(f"[CKSAAP{k}] 提取特征 …")
        train_df = build_feature_df(train_records, k)
        test_df  = build_feature_df(test_records, k)

        train_csv = OUT_ROOT / "train" / f"CKSAAP{k}.csv"
        test_csv  = OUT_ROOT / "test"  / f"CKSAAP{k}.csv"
        train_df.to_csv(train_csv, index=False, header=False)
        test_df.to_csv(test_csv,  index=False, header=False)
        print(f"  ✔ 保存 {train_csv} & {test_csv}")

# ========================= 4. 加列名 =========================

def add_headers():
    sub_dirs = ["train", "test"]
    for sub in sub_dirs:
        for k in K_SPACES:
            file_path = OUT_ROOT / sub / f"CKSAAP{k}.csv"
            df = pd.read_csv(file_path, header=None)
            exp_cols = 1 + 400 * (k + 1)
            if df.shape[1] != exp_cols:
                raise ValueError(f"{file_path} 维度不匹配")
            cols = ["target"] + [f"{pair}_{kk}" for kk in range(k + 1) for pair in PAIRS]
            df.columns = cols
            new_path = file_path.with_name(file_path.stem + "_with_header.csv")
            df.to_csv(new_path, index=False)
            print(f"  ✔ 列名添加完成: {new_path}")

# ========================= 5. 特征排序 =========================
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
import shap
from skrebate import ReliefF


def rank_features(csv_path: Path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values
    y = df["target"].values
    feats = df.columns[1:]

    # ---- SHAP (XGBoost) ----
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric="logloss")
    xgb.fit(Xs, y)
    explainer = shap.TreeExplainer(xgb)
    shap_vals = explainer.shap_values(Xs)
    shap_rank = np.abs(shap_vals).mean(0)
    shap_df = pd.DataFrame({"feature": feats, "shap_value": shap_rank}).sort_values("shap_value", ascending=False)
    shap_df.to_csv(csv_path.with_stem(csv_path.stem.replace("_with_header", "_shap")), index=False)

    # ---- ReliefF ----
    rel = ReliefF(n_neighbors=10, n_features_to_select=len(feats))
    rel.fit(Xs, y)
    rel_df = pd.DataFrame({"feature": feats, "relief_score": rel.feature_importances_}).sort_values("relief_score", ascending=False)
    rel_df.to_csv(csv_path.with_stem(csv_path.stem.replace("_with_header", "_relief")), index=False)

    # ---- ANOVA ----
    f_val, _ = f_classif(X, y)
    anova_df = pd.DataFrame({"feature": feats, "f_score": f_val}).sort_values("f_score", ascending=False)
    anova_df.to_csv(csv_path.with_stem(csv_path.stem.replace("_with_header", "_anova")), index=False)
    print(f"  ✔ 排序完成: {csv_path.stem}")


def run_ranking():
    for k in K_SPACES:
        csv_path = OUT_ROOT / "train" / f"CKSAAP{k}_with_header.csv"
        rank_features(csv_path)

# ========================= main =========================
if __name__ == "__main__":
    prepare_cksaap_files()
    add_headers()
    run_ranking()
    print("=== 全流程完成 ===")
