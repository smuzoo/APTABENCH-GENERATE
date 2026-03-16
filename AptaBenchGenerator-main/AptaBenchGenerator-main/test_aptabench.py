import sys
sys.path.append('aptamer_model')

from src.predictor import AptamerLigandPredictor

# Initialize the model
model = AptamerLigandPredictor()

# Test single prediction
seq = "GGGAGAATTCCCGCGGCAGAAGCCCACCTGGCTTTGAACTCTATGTTATTGGGTGGGGGAAACTTAAGAAAACTACCACCCTTCAACATTACCGCCCTTCAGCCTGCCAGCGCCCTGCAGCCCGGGAAGCTT"
smiles = "Nc1c(S(=O)(=O)O)cc(Nc2ccc(Nc3nc(Cl)nc(Nc4ccccc4S(=O)(=O)O)n3)c(S(=O)(=O)O)c2)c2c1C(=O)c1ccccc1C2=O"

proba = model.predict_proba(seq, smiles)
label = model.predict(seq, smiles)

print("Probability of binding: {}".format(proba))
print("Predicted label: {}".format(label))

# Test batch prediction
sequences = ["ACGT", "GGCC"]
smiles_list = ["CCO", "c1ccccc1"]

proba_batch = model.predict_proba_batch(sequences, smiles_list)
labels_batch = model.predict_batch(sequences, smiles_list)

print("Batch probabilities: {}".format(list(proba_batch)))
print("Batch labels: {}".format(list(labels_batch)))</content>
<parameter name="filePath">c:\Users\smuzoo\Documents\DNK NANOMACHINES\AptaBenchGenerator-main\AptaBenchGenerator-main\test_aptabench.py