"""Retrain the AptaBench model ONLY on glyphosate-related data.

This script reads glyphosate interaction data and trains a new LightGBM model
using ONLY these examples (no original AptaBench data).

Usage:
  python retrain_only_glyphosate.py \
    --glyphosate-interactions "../../NEW MODEL/glyphosate_interactions_for_model.csv" \
    --glyphosate-sequences "../../NEW MODEL/glyphosate_sequences_master.csv" \
    --glyphosate-raw "../../NEW MODEL/glyphosate_records_raw.csv" \
    --output-model "models/lgbm_model_only_glyphosate.txt" \
    --output-csv "data/glyphosate_only_dataset.csv"
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
        description="Retrain AptaBench model ONLY on glyphosate data."
    )
    parser.add_argument(
        "--glyphosate-interactions",
        type=Path,
        default="../../NEW MODEL/glyphosate_interactions_for_model.csv",
        help="Path to glyphosate interactions CSV.",
    )
    parser.add_argument(
        "--glyphosate-sequences",
        type=Path,
        default="../../NEW MODEL/glyphosate_sequences_master.csv",
        help="Path to glyphosate sequences CSV.",
    )
    parser.add_argument(
        "--glyphosate-raw",
        type=Path,
        default="../../NEW MODEL/glyphosate_records_raw.csv",
        help="Path to raw glyphosate records CSV.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default="aptamer_model/models/lgbm_model_only_glyphosate.txt",
        help="Path to save the retrained model.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default="aptamer_model/data/glyphosate_only_dataset.csv",
        help="Path to save the combined dataset.",
    )
    return parser.parse_args()


def load_glyphosate_data(interactions_path, sequences_path, raw_path):
    """Load and merge glyphosate data."""
    interactions_df = pd.read_csv(interactions_path)
    sequences_df = pd.read_csv(sequences_path)
    raw_df = pd.read_csv(raw_path)

    # Merge interactions with sequences
    merged = interactions_df.merge(
        sequences_df[["sequence_id", "sequence_5to3"]].rename(columns={"sequence_5to3": "sequence"}), on="sequence_id", how="left"
    )

    # For missing sequences, try raw_df
    missing_seqs = merged[merged["sequence"].isna()]
    if not missing_seqs.empty:
        raw_merged = missing_seqs.merge(
            raw_df[["sequence_id", "sequence_5to3"]], on="sequence_id", how="left"
        )
        merged.loc[merged["sequence"].isna(), "sequence"] = raw_merged["sequence_5to3"]

    # Add SMILES (glyphosate)
    merged["canonical_smiles"] = "Nc1c(S(=O)(=O)O)cc(Nc2ccc(Nc3nc(Cl)nc(Nc4ccccc4S(=O)(=O)O)n3)c(S(=O)(=O)O)c2)c2c1C(=O)c1ccccc1C2=O"

    # Normalize labels
    def normalize_label(label):
        if pd.isna(label):
            return 0
        try:
            num_label = float(label)
            return 1 if num_label == 1.0 else 0
        except (ValueError, TypeError):
            label = str(label).lower().strip()
            if label in POSITIVE_LABELS:
                return 1
            elif label in NEGATIVE_LABELS:
                return 0
            else:
                warnings.warn(f"Unknown label: {label}, treating as negative")
                return 0

    merged["label"] = merged["activity_label_binary"].apply(normalize_label)

    # Select columns
    final_df = merged[["sequence", "canonical_smiles", "label"]].dropna()

    return final_df


def main():
    args = parse_args()

    # Load glyphosate data
    glyphosate_df = load_glyphosate_data(
        args.glyphosate_interactions, args.glyphosate_sequences, args.glyphosate_raw
    )

    print(f"Loaded {len(glyphosate_df)} glyphosate examples.")
    print(glyphosate_df["label"].value_counts())

    # Save dataset
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    glyphosate_df.to_csv(args.output_csv, index=False)
    print(f"Saved dataset to {args.output_csv}")

    # Train model
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    X, y = train.build_features(glyphosate_df)
    clf = train.train_model(X, y)
    train.save_model(clf, str(args.output_model))
    print(f"Saved model to {args.output_model}")


if __name__ == "__main__":
    main()