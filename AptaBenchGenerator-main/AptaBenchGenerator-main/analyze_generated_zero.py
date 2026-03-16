"""Analyze generated aptamers, with focus on label=0 (low predicted binding).

This script helps you inspect whether the model is reliably distinguishing "bad"
sequences and what those sequences look like.

Usage:
    python analyze_generated_zero.py
"""

import collections
import re

import pandas as pd


def gc_content(seq: str) -> float:
    seq = seq.upper()
    if not seq:
        return 0.0
    g = seq.count("G")
    c = seq.count("C")
    return (g + c) / len(seq)


def longest_homopolymer(seq: str) -> int:
    # Maximum run length of the same character (A/C/G/T/U)
    if not seq:
        return 0
    max_run = 1
    run = 1
    for a, b in zip(seq, seq[1:]):
        if a == b:
            run += 1
        else:
            max_run = max(max_run, run)
            run = 1
    return max(max_run, run)


def is_valid_seq(seq: str) -> bool:
    return bool(re.fullmatch(r"[ACGTUacgtu]+", seq))


def summarize_sequences(seqs):
    lengths = [len(s) for s in seqs]
    gcs = [gc_content(s) for s in seqs]
    homos = [longest_homopolymer(s) for s in seqs]

    return {
        "count": len(seqs),
        "len_min": min(lengths) if lengths else 0,
        "len_max": max(lengths) if lengths else 0,
        "len_mean": sum(lengths) / len(lengths) if lengths else 0,
        "gc_mean": sum(gcs) / len(gcs) if gcs else 0,
        "homopolymer_mean": sum(homos) / len(homos) if homos else 0,
    }


def format_summary(name, summary):
    return (
        f"{name}: count={summary['count']} "
        f"len=[{summary['len_min']}..{summary['len_max']}] mean={summary['len_mean']:.1f} "
        f"gc_mean={summary['gc_mean']:.2f} homopolymer_mean={summary['homopolymer_mean']:.1f}"
    )


def main():
    path = "all_generated_aptamers.csv"
    df = pd.read_csv(path)

    print(f"Total generated rows: {len(df)}")
    print(f"label=1: {int((df['label'] == 1).sum())}, label=0: {int((df['label'] == 0).sum())}\n")

    df0 = df[df['label'] == 0].copy()
    df1 = df[df['label'] == 1].copy()

    print("=== label=0 (low-proba) summary ===")
    print(f"proba min={df0['proba'].min():.4f}, mean={df0['proba'].mean():.4f}, median={df0['proba'].median():.4f}")
    print("Top 10 lowest-proba sequences:")
    for _, row in df0.nsmallest(10, 'proba').iterrows():
        print(f"  proba={row['proba']:.4f}, seq={row['sequence']}")

    print("\n=== label=1 (high-proba) summary ===")
    print(f"proba min={df1['proba'].min():.4f}, mean={df1['proba'].mean():.4f}, median={df1['proba'].median():.4f}")
    print("Top 10 highest-proba sequences:")
    for _, row in df1.nlargest(10, 'proba').iterrows():
        print(f"  proba={row['proba']:.4f}, seq={row['sequence']}")

    # Compare sequence statistics
    seqs0 = df0['sequence'].astype(str).tolist()
    seqs1 = df1['sequence'].astype(str).tolist()

    print("\n=== sequence stats ===")
    print(format_summary('label=0', summarize_sequences(seqs0)))
    print(format_summary('label=1', summarize_sequences(seqs1)))

    # Check validity of sequences
    invalid0 = [s for s in seqs0 if not is_valid_seq(s)]
    invalid1 = [s for s in seqs1 if not is_valid_seq(s)]
    print(f"\nInvalid sequences (non-ACGTU) count: label=0 {len(invalid0)}, label=1 {len(invalid1)}")

    if invalid0:
        print("Example invalid label=0 sequence:", invalid0[0])
    if invalid1:
        print("Example invalid label=1 sequence:", invalid1[0])


if __name__ == "__main__":
    main()
