"""Train LGBMClassifier for aptamer–ligand classification."""

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from .features import onehot_with_type_bit, morgan_fp
from .utils import get_data_path, get_models_path


LGBM_PARAMS = {
    "n_estimators": 244,
    "learning_rate": 0.12081607580760213,
    "num_leaves": 103,
    "max_depth": 11,
    "subsample": 0.6683011365580269,
    "colsample_bytree": 0.9291870958994618,
    "reg_lambda": 0.0319163744440175,
}


def load_dataset(data_path=None):
    """Load CSV dataset; drop rows with missing label/sequence/smiles."""
    if data_path is None:
        data_path = get_data_path("AptaBench_dataset_v2.csv")
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    required = ["sequence", "canonical_smiles", "label"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


def build_features(df):
    """Build X from sequence (onehot) and SMILES (Morgan FP), y from label."""
    seqs = df["sequence"].fillna("").tolist()
    smiles_list = df["canonical_smiles"].fillna("").tolist()
    X_seq = onehot_with_type_bit(seqs)
    X_mol = morgan_fp(smiles_list)
    X = np.hstack([X_seq, X_mol.astype(np.float64)])
    y = df["label"].values
    return X, y


def train_model(X, y, model_dir=None):
    """Train LGBMClassifier and return fitted model."""
    if model_dir is None:
        model_dir = get_models_path().parent
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    clf = lgb.LGBMClassifier(**LGBM_PARAMS, verbose=-1, random_state=42)
    clf.fit(X, y)
    return clf


def save_model(clf, model_path=None):
    """Save LightGBM booster to text file."""
    if model_path is None:
        model_path = get_models_path("lgbm_model.txt")
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    clf.booster_.save_model(str(model_path))


def main():
    data_path = get_data_path("AptaBench_dataset_v2.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Place AptaBench_dataset_v2.csv in data/.")
    df = load_dataset(data_path)
    X, y = build_features(df)
    clf = train_model(X, y)
    save_model(clf, get_models_path("lgbm_model.txt"))
    proba = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    print(f"Train ROC-AUC: {auc:.4f}")
    print(f"Model saved to {get_models_path('lgbm_model.txt')}")


if __name__ == "__main__":
    main()
