import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

############################
# 1) 定义 TransformerModel #
############################
class TransformerModel(nn.Module):
    """A minimal single-layer Transformer encoder followed by a linear head."""

    def __init__(self, input_dim: int):
        super().__init__()
        # 使 nhead 能整除 input_dim；若不能整除则取 1
        nhead = next((h for h in range(8, 0, -1) if input_dim % h == 0), 1)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.fc = nn.Linear(input_dim, 1)  # 二分类 logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F)
        x = x.unsqueeze(1)  # (B, 1, F)
        x = self.transformer(x)  # (B, 1, F)
        x = x.squeeze(1)  # (B, F)
        return self.fc(x)  # (B, 1)


######################################################################
# 2) 训练 Transformer 并返回 (预测概率, 模型列表)                           #
######################################################################

def train_and_predict_transformer(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    *,
    input_dim: int,
    n_splits: int = 5,
    random_state: int = 42,
    epochs: int = 10,
    lr: float = 1e-3,
):
    """5 折交叉验证训练 Transformer。

    返回:
    -------
    probs_test_mean : (N_test, 2) numpy array
    fold_models     : List[nn.Module]  # 每折 1 个模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    test_preds, fold_models = [], []

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

        model = TransformerModel(input_dim).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_test = model(X_test_t)
            prob_class1 = torch.sigmoid(logits_test).cpu().numpy().flatten()
            prob_class0 = 1.0 - prob_class1
            test_preds.append(np.stack([prob_class0, prob_class1], axis=1))

        # 将模型搬到 CPU 方便保存，节省 GPU 显存
        fold_models.append(model.cpu())
        torch.cuda.empty_cache()

    probs_test_mean = np.mean(test_preds, axis=0)
    return probs_test_mean, fold_models


###############################################################
# 3) IFS 循环：按 AUC 仅保存最佳模型                               #
###############################################################

def incremental_feature_selection(
    *,
    train_csv: str,
    test_csv: str,
    shap_csv: str,
    relief_csv: str,
    anova_csv: str,
    output_root: str,
    dataset_name: str,
):
    """对单个数据集执行 IFS，并在每种特征排序方法下保存最佳 AUC 的模型。"""

    # 1) 准备目录
    dataset_dir = os.path.join(output_root, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # 2) 读取数据
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    X_train_full = df_train.iloc[:, 1:].values
    y_train_full = df_train["target"].values
    X_test_full = df_test.iloc[:, 1:].values
    y_test_full = df_test["target"].values  # 用于计算 AUC

    feature_names = list(df_train.columns[1:])

    # 3) 读取排序文件
    rank_dfs = {
        "shap": pd.read_csv(shap_csv),
        "relief": pd.read_csv(relief_csv),
        "anova": pd.read_csv(anova_csv),
    }

    # 4) 针对每种排序方法进行 IFS
    for method, rdf in rank_dfs.items():
        method_dir = os.path.join(dataset_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        models_dir = os.path.join(method_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        best_auc = None  # 当前最佳 AUC

        ranked_feats = [f for f in rdf["feature"].tolist() if f in feature_names]
        num_feats_total = len(ranked_feats)

        # 辅助: 根据特征集合抽取数据
        idx_map = {name: i for i, name in enumerate(feature_names)}
        def subset(X, names):
            return X[:, [idx_map[n] for n in names]]

        for n in range(1, num_feats_total + 1):
            topn_feats = ranked_feats[:n]
            X_train_sub = subset(X_train_full, topn_feats)
            X_test_sub = subset(X_test_full, topn_feats)

            probs_test, fold_models = train_and_predict_transformer(
                X_train_sub,
                y_train_full,
                X_test_sub,
                input_dim=X_train_sub.shape[1],
                n_splits=5,
                epochs=10,
                lr=1e-3,
            )

            auc = roc_auc_score(y_test_full, probs_test[:, 1])
            csv_path = os.path.join(method_dir, f"top{n}_pred.csv")
            pd.DataFrame(probs_test, columns=["Score_0", "Score_1"]).to_csv(csv_path, index=False)

            print(f"[{dataset_name}][{method}] top-{n} features AUC={auc:.4f}")

            # 保存模型: 若无最佳或当前 AUC 更优
            if best_auc is None or auc > best_auc:
                best_auc = auc
                # 清空旧模型文件
                for f in os.listdir(models_dir):
                    os.remove(os.path.join(models_dir, f))

                # 保存当前 5 折模型
                for fold_idx, model in enumerate(fold_models):
                    model_path = os.path.join(models_dir, f"top{n}_fold{fold_idx}.pt")
                    torch.save(model.state_dict(), model_path)

                # 同步保存当前最佳特征数记录
                with open(os.path.join(method_dir, "best_info.txt"), "w") as fh:
                    fh.write(f"best_n={n}\nbest_auc={best_auc:.6f}\n")


###############################################
# 4) 主函数：依次处理 CKSAAP4 与 CKSAAP5       #
###############################################
if __name__ == "__main__":

    base_dir = "data"
    output_root = "IFS_results"
    os.makedirs(output_root, exist_ok=True)

    datasets = {
        "CKSAAP4": {
            "train": os.path.join(base_dir, "train", "CKSAAP4_with_header.csv"),
            "test": os.path.join(base_dir, "test", "CKSAAP4_with_header.csv"),
            "shap": os.path.join(base_dir, "train", "CKSAAP4_shap.csv"),
            "relief": os.path.join(base_dir, "train", "CKSAAP4_relief.csv"),
            "anova": os.path.join(base_dir, "train", "CKSAAP4_anova.csv"),
        },
        "CKSAAP5": {
            "train": os.path.join(base_dir, "train", "CKSAAP5_with_header.csv"),
            "test": os.path.join(base_dir, "test", "CKSAAP5_with_header.csv"),
            "shap": os.path.join(base_dir, "train", "CKSAAP5_shap.csv"),
            "relief": os.path.join(base_dir, "train", "CKSAAP5_relief.csv"),
            "anova": os.path.join(base_dir, "train", "CKSAAP5_anova.csv"),
        },
    }

    for name, paths in datasets.items():
        incremental_feature_selection(
            train_csv=paths["train"],
            test_csv=paths["test"],
            shap_csv=paths["shap"],
            relief_csv=paths["relief"],
            anova_csv=paths["anova"],
            output_root=output_root,
            dataset_name=name,
        )

    print("=== All Done! ===")