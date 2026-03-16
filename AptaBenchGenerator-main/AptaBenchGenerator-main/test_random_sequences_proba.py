"""Test: evaluate the LightGBM scoring model on random sequences.

This script generates random A/C/G/T sequences of a fixed length and
runs them through the existing AptamerLigandPredictor to see the
probability distribution.

Run with:
    python test_random_sequences_proba.py

If the predicted probabilities are mostly ~1.0, then the predictor is likely
biased/overly optimistic on arbitrary sequences.
"""

import random
import sys

sys.path.append('aptamer_model')

from src.predictor import AptamerLigandPredictor


def random_seq(length=50):
    return ''.join(random.choice('ACGT') for _ in range(length))


def main(num_seqs=200, length=50, target_smiles=None):
    predictor = AptamerLigandPredictor()

    # Use a valid SMILES string (same as default in LLMGenerator) so RDKit parsing succeeds.
    if target_smiles is None:
        target_smiles = (
            "Nc1c(S(=O)(=O)O)cc(Nc2ccc(Nc3nc(Cl)nc(Nc4ccccc4S(=O)(=O)O)n3)c(S(=O)(=O)O)c2)"
            "c2c1C(=O)c1ccccc1C2=O"
        )

    sequences = [random_seq(length) for _ in range(num_seqs)]

    probas = predictor.predict_proba_batch(sequences, [target_smiles] * num_seqs)

    proba_list = list(probas)
    mean_proba = sum(proba_list) / len(proba_list)
    sorted_proba = sorted(proba_list)

    for threshold in [0.5, 0.8, 0.9]:
        frac = sum(p > threshold for p in proba_list) / len(proba_list)
        print(f"proba > {threshold}: {frac:.3f}")

    print("\nSummary:")
    print(f"count: {len(proba_list)}")
    print(f"mean: {mean_proba:.4f}")
    print(f"median: {sorted_proba[len(sorted_proba)//2]:.4f}")
    print(f"25%: {sorted_proba[len(sorted_proba)//4]:.4f}")
    print(f"75%: {sorted_proba[(len(sorted_proba)*3)//4]:.4f}")


if __name__ == "__main__":
    main()
