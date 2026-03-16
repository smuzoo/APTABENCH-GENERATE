from aptamer_model.src.predictor import AptamerLigandPredictor

# Initialize the model
model = AptamerLigandPredictor()

# Test single prediction
seq = "GGGAGAATTCCCGCGGCAGAAGCCCACCTGGCTTTGAACTCTATGTTATTGGGTGGGGGAAACTTAAGAAAACTACCACCCTTCAACATTACCGCCCTTCAGCCTGCCAGCGCCCTGCAGCCCGGGAAGCTT"
smiles = "Nc1c(S(=O)(=O)O)cc(Nc2ccc(Nc3nc(Cl)nc(Nc4ccccc4S(=O)(=O)O)n3)c(S(=O)(=O)O)c2)c2c1C(=O)c1ccccc1C2=O"

proba = model.predict_proba(seq, smiles)
label = model.predict(seq, smiles)

print(f"Probability of binding: {proba}")
print(f"Predicted label: {label}")

# Test batch prediction
sequences = ["ACGT", "GGCC"]
smiles_list = ["CCO", "c1ccccc1"]

proba_batch = model.predict_proba_batch(sequences, smiles_list)
labels_batch = model.predict_batch(sequences, smiles_list)

print(f"Batch probabilities: {proba_batch}")
print(f"Batch labels: {labels_batch}")</content>
<parameter name="filePath">c:\Users\smuzoo\Documents\DNK NANOMACHINES\test_aptabench.py