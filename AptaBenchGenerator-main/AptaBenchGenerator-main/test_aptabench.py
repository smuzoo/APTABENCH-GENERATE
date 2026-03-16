import sys
sys.path.append('aptamer_model')

try:
    from src.predictor import AptamerLigandPredictor
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: {}.\nPlease install requirements via `pip install -r aptamer_model/requirements.txt`".format(e.name)
    )


def test_predictor_basic_behavior():
    """Basic sanity checks for the AptamerLigandPredictor."""

    model = AptamerLigandPredictor()

    # A realistic aptamer + SMILES pair (should not crash)
    seq = (
        "GGGAGAATTCCCGCGGCAGAAGCCCACCTGGCTTTGAACTCTATGTTATTGGGTGGGGGAAACTTAAGAAAACTACCACCCTTCAACATTACCGCCCTTCAGCCTGCCAGCGCCCTGCAGCCCGGGAAGCTT"
    )
    smiles = "Nc1c(S(=O)(=O)O)cc(Nc2ccc(Nc3nc(Cl)nc(Nc4ccccc4S(=O)(=O)O)n3)c(S(=O)(=O)O)c2)c2c1C(=O)c1ccccc1C2=O"

    proba = model.predict_proba(seq, smiles)
    assert isinstance(proba, float), "predict_proba must return float"
    assert 0.0 <= proba <= 1.0, "probability must be in [0, 1]"

    label = model.predict(seq, smiles)
    assert label in (0, 1), "predict must return 0 or 1"

    # Batch prediction matches input length and returns valid probabilities
    sequences = ["ACGT", "GGCC"]
    smiles_list = ["CCO", "c1ccccc1"]

    proba_batch = model.predict_proba_batch(sequences, smiles_list)
    labels_batch = model.predict_batch(sequences, smiles_list)

    assert len(proba_batch) == len(sequences)
    assert len(labels_batch) == len(sequences)
    assert all(0.0 <= p <= 1.0 for p in proba_batch)
    assert all(l in (0, 1) for l in labels_batch)

    # Edge cases: empty inputs should not crash and should return empty arrays
    assert model.predict_proba_batch([], []) == []
    assert model.predict_batch([], []) == []


if __name__ == "__main__":
    # Run the test function directly for environments without pytest.
    test_predictor_basic_behavior()
    print("Basic predictor behavior test passed.")
