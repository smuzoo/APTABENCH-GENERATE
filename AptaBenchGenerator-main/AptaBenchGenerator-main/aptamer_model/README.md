## Aptamer–Small Molecule Classification

This repository provides a ready-to-use LightGBM model for **aptamer–small molecule binding classification** (binary label 0/1) and a simple Python API for inference.

### Inference

The main interface is the `AptamerLigandPredictor` class. It loads a trained LightGBM booster from `models/lgbm_model.txt` by default and exposes single-sample and batch prediction methods.

```python
from src.predictor import AptamerLigandPredictor

model = AptamerLigandPredictor()  # or AptamerLigandPredictor(model_path="path/to/lgbm_model.txt")

# Single prediction
seq = "ACGTACGT"
smiles = "CCO"
proba = model.predict_proba(seq, smiles)          # float, P(label=1)
label = model.predict(seq, smiles)               # int, 0 or 1
```

- `predict_proba(sequence, smiles)` – returns probability of class 1 as `float`.
- `predict(sequence, smiles, threshold=0.5)` – returns predicted label `0` or `1`.

#### Batch inference

For batch scoring, pass lists of equal length:

```python
sequences = ["ACGT", "ACGU", "GGCC"]
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]

proba_batch = model.predict_proba_batch(sequences, smiles_list)  # np.ndarray of probabilities
labels_batch = model.predict_batch(sequences, smiles_list, threshold=0.5)  # np.ndarray of 0/1
```

- `predict_proba_batch(sequences, smiles_list)` – returns a NumPy array of probabilities for class 1.
- `predict_batch(sequences, smiles_list, threshold=0.5)` – returns a NumPy array of predicted labels (0/1).

### Installation

Install dependencies into your current Python environment:

```bash
pip install -r requirements.txt
```

