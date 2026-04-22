"""Evaluate sequence candidates with the AptaBench model(s).

This script scores candidate sequences using the retrained glyphosate model
and optionally the original AptaBench model. It also computes simple sequence
features like GC content, length, and Shannon entropy.

Example:
  python evaluate_candidates.py \
    --input-csv "c:/Users/smuzoo/Documents/DNK NANOMACHINES/NEW MODEL/candidate_sequences.csv" \
    --sequence-column sequence \
    --target-smiles "O=C(O)CNCP(=O)(O)O" \
    --model-path aptamer_model/models/lgbm_model_glyphosate_retrained_test.txt \
    --old-model-path aptamer_model/models/lgbm_model.txt \
    --output-csv evaluated_candidates.csv
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.append("aptamer_model")
from src.predictor import AptamerLigandPredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score candidate sequences with the AptaBench model(s)."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV file containing candidate sequences.",
    )
    parser.add_argument(
        "--sequence-column",
        default="sequence",
        help="Column name containing nucleotide sequences.",
    )
    parser.add_argument(
        "--smiles-column",
        default="canonical_smiles",
        help="Column name containing SMILES strings for the target.",
    )
    parser.add_argument(
        "--target-smiles",
        default="O=C(O)CNCP(=O)(O)O",
        help="Fallback target SMILES if column is missing or empty.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("aptamer_model/models/lgbm_model_glyphosate_retrained_test.txt"),
        help="Path to the retrained model file.",
    )
    parser.add_argument(
        "--old-model-path",
        type=Path,
        default=None,
        help="Optional path to the original model for comparison.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("evaluated_candidates.csv"),
        help="Path to save the scored candidate table.",
    )
    return parser.parse_args()


def gc_content(sequence: str) -> float:
    sequence = (sequence or "").upper()
    if not sequence:
        return 0.0
    gc = sequence.count("G") + sequence.count("C")
    return gc / len(sequence)


def shannon_entropy(sequence: str) -> float:
    sequence = (sequence or "").upper()
    if not sequence:
        return 0.0
    counts = {}
    for base in sequence:
        counts[base] = counts.get(base, 0) + 1
    entropy = 0.0
    for count in counts.values():
        p = count / len(sequence)
        entropy -= p * math.log2(p)
    return entropy


def evaluate_sequences(df: pd.DataFrame, seq_col: str, smi_col: str, default_smiles: str,
                       model_path: Path, old_model_path: Path | None = None) -> pd.DataFrame:
    predictor = AptamerLigandPredictor(model_path)
    old_predictor = None
    if old_model_path:
        old_predictor = AptamerLigandPredictor(old_model_path)

    sequences = df[seq_col].fillna("").astype(str).tolist()
    smiles = []
    for idx, row in df.iterrows():
        value = row.get(smi_col, None)
        if pd.isna(value) or str(value).strip() == "":
            smiles.append(default_smiles)
        else:
            smiles.append(str(value).strip())

    df = df.copy()
    df["length"] = [len(s) for s in sequences]
    df["gc_fraction"] = [gc_content(s) for s in sequences]
    df["shannon_entropy"] = [shannon_entropy(s) for s in sequences]
    df["proba_new_model"] = predictor.predict_proba_batch(sequences, smiles)
    if old_predictor is not None:
        df["proba_old_model"] = old_predictor.predict_proba_batch(sequences, smiles)
    return df


def main():
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Retrained model not found: {args.model_path}")
    if args.old_model_path and not args.old_model_path.exists():
        raise FileNotFoundError(f"Old model not found: {args.old_model_path}")

    df = pd.read_csv(args.input_csv, dtype=str)
    if args.sequence_column not in df.columns:
        raise ValueError(f"Sequence column '{args.sequence_column}' not found in input CSV.")

    evaluated = evaluate_sequences(
        df,
        seq_col=args.sequence_column,
        smi_col=args.smiles_column,
        default_smiles=args.target_smiles,
        model_path=args.model_path,
        old_model_path=args.old_model_path,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(args.output_csv, index=False)
    print(f"Saved evaluated candidates to {args.output_csv}")
    print("Summary:")
    print(evaluated[["proba_new_model"]].describe())
    if args.old_model_path:
        print(evaluated[["proba_new_model", "proba_old_model"]].describe())


if __name__ == "__main__":
    main()
