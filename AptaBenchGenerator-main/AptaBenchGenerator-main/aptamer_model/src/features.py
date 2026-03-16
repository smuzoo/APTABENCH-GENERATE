"""Feature extraction for aptamer and ligand (SMILES) encodings."""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfg


def infer_types_from_sequences(seqs):
    """RNA if contains 'U', otherwise DNA."""
    types = []
    for s in seqs:
        if s is None:
            types.append("DNA")
        else:
            s = str(s).upper()
            types.append("RNA" if "U" in s else "DNA")
    return types


def onehot_with_type_bit(seqs, max_len=216):
    types = infer_types_from_sequences(seqs)
    alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    N = len(seqs)
    out = np.zeros((N, max_len * 4 + 1), dtype=np.float64)
    for i, (s, t) in enumerate(zip(seqs, types)):
        d = 1.0 if t == "RNA" else 0.0
        raw = (s or "").upper().replace("U", "T")
        raw = "".join(ch for ch in raw if ch in alphabet)[:max_len]
        for j, ch in enumerate(raw):
            out[i, j * 4 + alphabet[ch]] = 1.0
        out[i, -1] = d
    return out


def morgan_fp(smiles_list, n_bits=1024, radius=2, counts=False):
    X = np.zeros((len(smiles_list), n_bits), dtype=np.int32 if counts else np.uint8)
    gen = rfg.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi) if smi else "")
        if mol is None:
            continue
        if counts:
            fp = gen.GetCountFingerprint(mol)
            for idx, val in fp.GetNonzeroElements().items():
                if idx < n_bits:
                    X[i, idx] = val
        else:
            fp = gen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, X[i])
    return X
