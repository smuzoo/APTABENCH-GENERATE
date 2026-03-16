"""Inference API for aptamer–ligand classification."""

import numpy as np
import lightgbm as lgb

from .features import onehot_with_type_bit, morgan_fp
from .utils import get_project_root


# Default feature dimensions (must match train)
DEFAULT_MAX_LEN = 216
DEFAULT_N_BITS = 1024


class AptamerLigandPredictor:
    """Predictor for aptamer–small molecule binding (0/1 classification)."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = get_project_root() / "models" / "lgbm_model.txt"
        self._path = model_path
        self._booster = lgb.Booster(model_file=str(model_path))
        self._max_len = DEFAULT_MAX_LEN
        self._n_bits = DEFAULT_N_BITS

    def _build_features(self, sequence, smiles):
        """Build feature vector for one (sequence, smiles) pair. Returns shape (1, n_features)."""
        seq = sequence if sequence is not None else ""
        smi = smiles if smiles is not None else ""
        X_seq = onehot_with_type_bit([seq], max_len=self._max_len)
        X_mol = morgan_fp([smi], n_bits=self._n_bits)
        X = np.hstack([X_seq, X_mol])
        return X.astype(np.float64)

    def _build_features_batch(self, sequences, smiles_list):
        """Build feature matrix for (sequences, smiles_list). Returns shape (N, n_features)."""
        n = len(sequences)
        if n != len(smiles_list):
            raise ValueError("sequences and smiles_list must have the same length")
        seqs = [s if s is not None else "" for s in sequences]
        smis = [s if s is not None else "" for s in smiles_list]
        X_seq = onehot_with_type_bit(seqs, max_len=self._max_len)
        X_mol = morgan_fp(smis, n_bits=self._n_bits)
        X = np.hstack([X_seq, X_mol.astype(np.float64)])
        return X

    def predict_proba(self, sequence, smiles):
        """Return probability of class 1 (binding)."""
        X = self._build_features(sequence, smiles)
        proba = self._booster.predict(X)[0]
        if isinstance(proba, np.ndarray):
            return float(proba[1]) if proba.size > 1 else float(proba[0])
        return float(proba)

    def predict_proba_batch(self, sequences, smiles_list):
        """Return array of probabilities of class 1 for each (sequence, smiles) pair."""
        if len(sequences) == 0:
            return np.array([], dtype=np.float64)
        X = self._build_features_batch(sequences, smiles_list)
        pred = self._booster.predict(X)
        if isinstance(pred, np.ndarray) and pred.ndim > 1:
            return pred[:, 1] if pred.shape[1] > 1 else pred.ravel()
        return np.asarray(pred, dtype=np.float64)

    def predict(self, sequence, smiles, threshold=0.5):
        """Return predicted class 0 or 1."""
        p = self.predict_proba(sequence, smiles)
        return 1 if p >= threshold else 0

    def predict_batch(self, sequences, smiles_list, threshold=0.5):
        """Return array of predicted classes 0 or 1 for each (sequence, smiles) pair."""
        proba = self.predict_proba_batch(sequences, smiles_list)
        return (proba >= threshold).astype(np.int32)
