"""Build a candidate evaluation set from NEW MODEL data and score it with both models."""

import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parents[1]
NEW_MODEL = WORKSPACE_ROOT / "NEW MODEL"
sys.path.append(str(ROOT / "aptamer_model"))
from src.predictor import AptamerLigandPredictor


def load_candidate_set() -> pd.DataFrame:
    interactions = pd.read_csv(NEW_MODEL / "glyphosate_interactions_for_model.csv", dtype=str)
    sequences = pd.read_csv(NEW_MODEL / "glyphosate_sequences_master.csv", dtype=str)
    merged = interactions.merge(
        sequences[["sequence_id", "sequence_5to3"]], on="sequence_id", how="left"
    )

    positives = (
        merged[merged["target_name_normalized"] == "glyphosate"]
        [["sequence_5to3"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"sequence_5to3": "sequence"})
    )
    positives = positives.sample(min(15, len(positives)), random_state=1)
    counter = (
        merged[merged["target_name_normalized"].isin(["ampa", "glufosinate", "glycine", "alanine"])][["sequence_5to3", "target_name_normalized"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"sequence_5to3": "sequence", "target_name_normalized": "target"})
    )
    counter = counter.sample(min(15, len(counter)), random_state=1)
    random_seqs = [
        "".join(random.choice("ACGT") for _ in range(length))
        for length in random.choices([40, 50, 60, 70, 80], k=20)
    ]
    random_df = pd.DataFrame({"sequence": random_seqs, "target": ["glyphosate"] * len(random_seqs)})

    df = pd.concat(
        [
            positives.assign(target="glyphosate").assign(source="known_positive"),
            counter.assign(source="counter_target"),
            random_df.assign(source="random"),
        ],
        ignore_index=True,
    )
    df["canonical_smiles"] = "O=C(O)CNCP(=O)(O)O"
    return df


def score_candidates(df: pd.DataFrame):
    new_model = AptamerLigandPredictor(ROOT / "aptamer_model" / "models" / "lgbm_model_glyphosate_retrained_test.txt")
    old_model = AptamerLigandPredictor(ROOT / "aptamer_model" / "models" / "lgbm_model.txt")
    sequences = df["sequence"].fillna("").tolist()
    smiles = df["canonical_smiles"].fillna("O=C(O)CNCP(=O)(O)O").tolist()

    df = df.copy()
    df["proba_new_model"] = new_model.predict_proba_batch(sequences, smiles)
    df["proba_old_model"] = old_model.predict_proba_batch(sequences, smiles)
    return df


def main():
    candidates = load_candidate_set()
    candidates.to_csv(ROOT / "candidate_sequences_for_evaluation.csv", index=False)
    print("Candidate CSV written:", ROOT / "candidate_sequences_for_evaluation.csv")
    scored = score_candidates(candidates)
    scored.to_csv(ROOT / "evaluated_candidate_sequences.csv", index=False)
    print("Evaluated candidates saved:", ROOT / "evaluated_candidate_sequences.csv")
    print(scored.groupby("source")[['proba_new_model', 'proba_old_model']].mean())


if __name__ == "__main__":
    main()
