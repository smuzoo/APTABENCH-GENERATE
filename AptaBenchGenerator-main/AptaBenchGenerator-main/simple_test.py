import sys
sys.path.append('aptamer_model')

from src.predictor import AptamerLigandPredictor

model = AptamerLigandPredictor()

seq = "ACGT"
smiles = "CCO"

proba = model.predict_proba(seq, smiles)
print("Proba:", proba)