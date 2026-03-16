import google.genai as genai
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.predictor import AptamerLigandPredictor

class LLMGenerator:
    def __init__(self, api_key, target_smiles="Nc1c(S(=O)(=O)O)cc(Nc2ccc(Nc3nc(Cl)nc(Nc4ccccc4S(=O)(=O)O)n3)c(S(=O)(=O)O)c2)c2c1C(=O)c1ccccc1C2=O"):
        self.client = genai.Client(api_key=api_key)
        self.predictor = AptamerLigandPredictor()
        self.target_smiles = target_smiles
        self.examples = []  # Для few-shot learning

    def generate_sequences(self, num_sequences=10, max_length=50):
        prompt = f"""
        Generate {num_sequences} unique DNA or RNA aptamer sequences (using A, C, G, T/U) that could bind to the molecule with SMILES: {self.target_smiles}.
        Each sequence should be between 20-100 nucleotides long.
        Aptamers are short DNA or RNA molecules that fold into specific 3D structures to bind targets.

        Examples of good aptamers:
        - GGGAGAATTCCCGCGGCAGAAGCCCACCTGGCTTTGAACTCTATGTTATTGGGTGGGGGAAACTTAAGAAAACTACCACCCTTCAACATTACCGCCCTTCAGCCTGCCAGCGCCCTGCAGCCCGGGAAGCTT
        - GGGAAGGGAAGAAACUGCGGCUUCGGCCGGCUUCCC

        Output only the sequences, one per line, no extra text.
        """
        if self.examples:
            prompt += "\nPreviously successful examples:\n" + "\n".join(self.examples[:5])  # Топ 5

        response = self.client.models.generate_content(
            model='gemini-flash-latest',
            contents=prompt
        )
        sequences = [line.strip() for line in response.text.split('\n') if line.strip() and len(line.strip()) > 10]
        return sequences[:num_sequences]

    def evaluate_sequences(self, sequences):
        probas = self.predictor.predict_proba_batch(sequences, [self.target_smiles] * len(sequences))
        results = list(zip(sequences, probas))
        results.sort(key=lambda x: x[1], reverse=True)  # Сортировка по вероятности
        return results

    def generate_and_evaluate(self, num_sequences=10, iterations=3):
        best_sequences = []
        for i in range(iterations):
            print(f"Iteration {i+1}")
            sequences = self.generate_sequences(num_sequences)
            results = self.evaluate_sequences(sequences)
            top = results[:3]  # Топ 3
            best_sequences.extend([seq for seq, proba in top if proba > 0.5])  # Только хорошие
            self.examples.extend([seq for seq, proba in top])  # Добавляем в примеры
            print(f"Top sequences: {top}")
        return best_sequences

# Пример использования
if __name__ == "__main__":
    api_key = "AIzaSyDiV5pHf-3tXrYMH8edV8DQxtNUkYmz8lE"
    generator = LLMGenerator(api_key)
    best = generator.generate_and_evaluate(num_sequences=5, iterations=2)
    print("Best sequences:", best)