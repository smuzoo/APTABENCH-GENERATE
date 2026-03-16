"""Compare statistics between the real AptaBench dataset and the generated aptamers.

Usage:
    python compare_real_generated.py

This script prints basic distribution summaries for:
 - real dataset (AptaBench_dataset_v2.csv)
 - generated dataset (all_generated_aptamers.csv)

It is intended as a quick sanity check to help you see
how the generated sequences compare to the real dataset.
"""

import pandas as pd


def summarize_real_dataset(path):
    df = pd.read_csv(path)

    out = {}
    out["total"] = len(df)
    out["label_counts"] = df["label"].value_counts().to_dict()
    out["label_1_percent"] = float((df["label"] == 1).mean())
    out["label_0_percent"] = float((df["label"] == 0).mean())

    # pKd summary
    if "pKd_value" in df.columns:
        out["pKd_mean"] = float(df["pKd_value"].mean())
        out["pKd_median"] = float(df["pKd_value"].median())
        out["pKd_q25"] = float(df["pKd_value"].quantile(0.25))
        out["pKd_q75"] = float(df["pKd_value"].quantile(0.75))

    return out


def summarize_generated(path):
    df = pd.read_csv(path)

    out = {}
    out["total"] = len(df)
    out["label_counts"] = df["label"].value_counts().to_dict()
    out["label_1_percent"] = float((df["label"] == 1).mean())
    out["label_0_percent"] = float((df["label"] == 0).mean())

    out["proba_mean"] = float(df["proba"].mean())
    out["proba_median"] = float(df["proba"].median())
    out["proba_q25"] = float(df["proba"].quantile(0.25))
    out["proba_q75"] = float(df["proba"].quantile(0.75))
    out["proba_gt_0.5"] = float((df["proba"] > 0.5).mean())
    out["proba_gt_0.8"] = float((df["proba"] > 0.8).mean())
    out["proba_gt_0.9"] = float((df["proba"] > 0.9).mean())

    return out


def main():
    real_path = "aptamer_model/data/AptaBench_dataset_v2.csv"
    gen_path = "all_generated_aptamers.csv"

    real = summarize_real_dataset(real_path)
    gen = summarize_generated(gen_path)

    print("=== Real dataset (AptaBench_dataset_v2.csv) ===")
    print(f"Total rows: {real['total']}")
    print(f"Label 1 fraction: {real['label_1_percent']:.3f}")
    print(f"Label 0 fraction: {real['label_0_percent']:.3f}")
    if "pKd_mean" in real:
        print("pKd mean: {:.3f}, median: {:.3f}, 25%: {:.3f}, 75%: {:.3f}".format(
            real["pKd_mean"], real["pKd_median"], real["pKd_q25"], real["pKd_q75"]
        ))

    print("\n=== Generated dataset (all_generated_aptamers.csv) ===")
    print(f"Total rows: {gen['total']}")
    print(f"Label 1 fraction: {gen['label_1_percent']:.3f}")
    print(f"Label 0 fraction: {gen['label_0_percent']:.3f}")
    print("proba mean: {:.3f}, median: {:.3f}, 25%: {:.3f}, 75%: {:.3f}".format(
        gen["proba_mean"], gen["proba_median"], gen["proba_q25"], gen["proba_q75"]
    ))
    print(
        "proba >0.5: {:.3f}, >0.8: {:.3f}, >0.9: {:.3f}".format(
            gen["proba_gt_0.5"], gen["proba_gt_0.8"], gen["proba_gt_0.9"]
        )
    )


if __name__ == "__main__":
    main()
