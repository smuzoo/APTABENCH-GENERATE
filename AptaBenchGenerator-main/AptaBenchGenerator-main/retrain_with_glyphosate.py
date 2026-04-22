"""Retrain the AptaBench model with glyphosate-related data.

This script reads user-provided interaction metadata from the NEW MODEL folder,
joins it to sequence text and ligand SMILES, merges it with the original
AptaBench dataset, and trains a new LightGBM model.

Usage:
  python retrain_with_glyphosate.py \
    --glyphosate-interactions "../../NEW MODEL/glyphosate_interactions_for_model.csv" \
    --glyphosate-sequences "../../NEW MODEL/glyphosate_sequences_master.csv" \
    --glyphosate-raw "../../NEW MODEL/glyphosate_records_raw.csv" \
    --output-model "models/lgbm_model_glyphosate_retrained.txt" \
    --output-csv "data/AptaBench_dataset_v2_with_glyphosate.csv"
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from aptamer_model.src import train


POSITIVE_LABELS = {
    "positive",
    "strong_positive",
    "weak_positive",
    "weak_positive_claim",
    "specificity_positive",
    "counter_target_positive",
    "selection_positive",
    "direct_affinity",
}
NEGATIVE_LABELS = {
    "negative",
    "specificity_negative",
    "counter_target_negative",
    "cross_reactivity_warning",
    "counter_target_uncertain",
    "uncertain",
    "no_binding",
    "specificity_negative",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain AptaBench model with glyphosate-related examples."
    )
    parser.add_argument(
        "--glyphosate-interactions",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "NEW MODEL" / "glyphosate_interactions_for_model.csv",
        help="Path to glyphosate interaction metadata CSV.",
    )
    parser.add_argument(
        "--glyphosate-sequences",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "NEW MODEL" / "glyphosate_sequences_master.csv",
        help="Path to deduplicated glyphosate sequence master CSV.",
    )
    parser.add_argument(
        "--glyphosate-raw",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "NEW MODEL" / "glyphosate_records_raw.csv",
        help="Path to raw glyphosate evidence CSV, used to map targets to SMILES.",
    )
    parser.add_argument(
        "--apta-bench-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "aptamer_model" / "data" / "AptaBench_dataset_v2.csv",
        help="Path to the original AptaBench dataset CSV.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path(__file__).resolve().parent / "aptamer_model" / "models" / "lgbm_model_glyphosate_retrained.txt",
        help="Path to save the retrained LightGBM model.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "aptamer_model" / "data" / "AptaBench_dataset_v2_with_glyphosate.csv",
        help="Path to save the combined training CSV.",
    )
    return parser.parse_args()


def build_smiles_map(raw_df: pd.DataFrame) -> dict:
    """Build a target name -> ligand SMILES mapping from raw data."""
    mapping = {}
    for _, row in raw_df.iterrows():
        target_norm = str(row.get("target_name_normalized", "")).strip()
        ligand_smiles = str(row.get("ligand_smiles", "")).strip()
        if not target_norm or not ligand_smiles or ligand_smiles.lower() in {"nan", "none"}:
            continue
        if target_norm in mapping and mapping[target_norm] != ligand_smiles:
            warnings.warn(
                f"Different SMILES values found for target '{target_norm}': "
                f"'{mapping[target_norm]}' != '{ligand_smiles}'. Using first value."
            )
            continue
        mapping[target_norm] = ligand_smiles

    if not mapping:
        raise ValueError("No SMILES mapping found in raw data.")

    return mapping


def normalize_label(raw_label):
    if pd.isna(raw_label):
        return None
    if isinstance(raw_label, (int, np.integer)):
        return int(raw_label)
    try:
        raw_label = str(raw_label).strip().lower()
    except Exception:
        return None
    if raw_label == "":
        return None
    if raw_label in {"0", "0.0", "false", "negative"}:
        return 0
    if raw_label in {"1", "1.0", "true", "positive"}:
        return 1
    if raw_label in POSITIVE_LABELS:
        return 1
    if raw_label in NEGATIVE_LABELS:
        return 0
    if "positive" in raw_label and "negative" not in raw_label:
        return 1
    if "negative" in raw_label or "uncertain" in raw_label or "warning" in raw_label:
        return 0
    return None


def load_glyphosate_examples(interactions_path: Path, sequences_path: Path, raw_path: Path) -> pd.DataFrame:
    interactions = pd.read_csv(interactions_path, dtype=str)
    sequences = pd.read_csv(sequences_path, dtype=str)
    raw = pd.read_csv(raw_path, dtype=str)

    if "sequence_id" not in interactions.columns:
        raise ValueError("glyphosate interactions file must contain 'sequence_id'.")
    if "sequence_id" not in sequences.columns:
        raise ValueError("glyphosate sequences master file must contain 'sequence_id'.")
    if "sequence_5to3" not in sequences.columns:
        raise ValueError("glyphosate sequences master file must contain 'sequence_5to3'.")

    smiles_map = build_smiles_map(raw)

    interactions = interactions.rename(columns={"target_name_normalized": "target_name"})
    interaction_sequences = interactions.merge(
        sequences[["sequence_id", "sequence_5to3"]], on="sequence_id", how="left", validate="many_to_one"
    )

    if interaction_sequences["sequence_5to3"].isna().any():
        raw_sequences = raw[["sequence_id", "sequence_5to3"]].drop_duplicates(subset=["sequence_id"])
        interaction_sequences = interaction_sequences.merge(
            raw_sequences,
            on="sequence_id",
            how="left",
            suffixes=("", "_raw"),
        )
        missing_seq = interaction_sequences["sequence_5to3"].isna()
        interaction_sequences.loc[missing_seq, "sequence_5to3"] = interaction_sequences.loc[missing_seq, "sequence_5to3_raw"]
        interaction_sequences = interaction_sequences.drop(columns=["sequence_5to3_raw"])

    missing_seq = interaction_sequences["sequence_5to3"].isna()
    if missing_seq.any():
        raise ValueError(
            f"Missing sequence text for {missing_seq.sum()} interaction rows. "
            "Check glyphosate_sequences_master.csv, glyphosate_records_raw.csv, and sequence_id values."
        )

    interaction_sequences["canonical_smiles"] = interaction_sequences["target_name"].map(smiles_map)
    missing_smiles = interaction_sequences[interaction_sequences["canonical_smiles"].isna()]
    if not missing_smiles.empty:
        missing_targets = sorted(set(missing_smiles["target_name"].dropna().unique()))
        raise ValueError(
            "Could not resolve SMILES for target names: "
            + ", ".join(missing_targets)
        )

    interaction_sequences["label"] = interaction_sequences.apply(
        lambda row: normalize_label(row.get("activity_label_binary"))
        if pd.notna(row.get("activity_label_binary"))
        else normalize_label(row.get("activity_label_multiclass"))
        if pd.notna(row.get("activity_label_multiclass"))
        else normalize_label(row.get("activity_label"))
        if pd.notna(row.get("activity_label"))
        else normalize_label(row.get("activity_label_granular")),
        axis=1,
    )

    result = interaction_sequences[
        ["sequence_id", "sequence_5to3", "canonical_smiles", "label", "target_name", "target_role", "activity_label_binary", "activity_label_multiclass"]
    ].rename(columns={"sequence_5to3": "sequence"})

    result = result.dropna(subset=["sequence", "canonical_smiles", "label"]).reset_index(drop=True)
    result["label"] = result["label"].astype(int)

    return result


def summarize_dataset(df: pd.DataFrame, label_name: str = "label") -> None:
    print(f"Total rows: {len(df)}")
    print(df[label_name].value_counts(dropna=False).to_string())
    if "target_name" in df.columns:
        print("Target counts:")
        print(df.groupby("target_name")[label_name].count().sort_values(ascending=False).to_string())


def main():
    args = parse_args()

    print("Loading AptaBench dataset from:", args.apta_bench_csv)
    apta_df = train.load_dataset(args.apta_bench_csv)
    print("Original AptaBench dataset summary:")
    summarize_dataset(apta_df)
    print()

    print("Loading glyphosate examples from:", args.glyphosate_interactions)
    glyphosate_df = load_glyphosate_examples(
        args.glyphosate_interactions,
        args.glyphosate_sequences,
        args.glyphosate_raw,
    )
    print("Glyphosate-related examples summary:")
    summarize_dataset(glyphosate_df)
    print()

    combined = pd.concat([apta_df, glyphosate_df[["sequence", "canonical_smiles", "label"]]], ignore_index=True)
    combined = combined.dropna(subset=["sequence", "canonical_smiles", "label"])
    combined = combined.drop_duplicates(subset=["sequence", "canonical_smiles", "label"])
    print("Combined dataset summary:")
    summarize_dataset(combined)
    print()

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output_csv, index=False)
        print(f"Saved combined training dataset to {args.output_csv}")

    print("Building features and training new model...")
    X, y = train.build_features(combined)
    clf = train.train_model(X, y, model_dir=args.output_model.parent)
    if args.output_model:
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        clf.booster_.save_model(str(args.output_model))
        print(f"Saved retrained model to {args.output_model}")

    proba = clf.predict_proba(X)[:, 1]
    auc = train.roc_auc_score(y, proba) if hasattr(train, "roc_auc_score") else None
    if auc is not None:
        print(f"Training ROC-AUC of retrained model: {auc:.4f}")
    else:
        print("Training completed. ROC-AUC score unavailable.")


if __name__ == "__main__":
    main()
