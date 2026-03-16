import google.genai as genai
import sys
import os
import math
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.predictor import AptamerLigandPredictor

class LLMGenerator:
    def __init__(self, api_key, target_smiles="Nc1c(S(=O)(=O)O)cc(Nc2ccc(Nc3nc(Cl)nc(Nc4ccccc4S(=O)(=O)O)n3)c(S(=O)(=O)O)c2)c2c1C(=O)c1ccccc1C2=O", prompt_version=1):
        self.client = genai.Client(api_key=api_key)
        self.predictor = AptamerLigandPredictor()
        self.target_smiles = target_smiles
        self.examples = []  # Для few-shot learning
        self.prompt_version = prompt_version

    def gc_content(self, seq: str) -> float:
        seq = seq.upper()
        if not seq:
            return 0.0
        g = seq.count("G")
        c = seq.count("C")
        return (g + c) / len(seq)

    def shannon_entropy(self, seq: str) -> float:
        counts = Counter(seq.upper())
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    def longest_homopolymer(self, seq: str) -> int:
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

    def generate_sequences(self, num_sequences=10, max_length=50):
        if self.prompt_version == 1:
            prompt = f"""
        Generate {num_sequences} unique DNA or RNA aptamer sequences (using A, C, G, T/U) that could bind to the molecule with SMILES: {self.target_smiles}.
        Each sequence should be between 20-100 nucleotides long.
        Aptamers are short DNA or RNA molecules that fold into specific 3D structures to bind targets.

        Examples of good aptamers:
        - GGGAGAATTCCCGCGGCAGAAGCCCACCTGGCTTTGAACTCTATGTTATTGGGTGGGGGAAACTTAAGAAAACTACCACCCTTCAACATTACCGCCCTTCAGCCTGCCAGCGCCCTGCAGCCCGGGAAGCTT
        - GGGAAGGGAAGAAACUGCGGCUUCGGCCGGCUUCCC

        Output only the sequences, one per line, no extra text.
        """
        elif self.prompt_version == 2:
            prompt = f"""
        Generate {num_sequences} unique DNA or RNA aptamer sequences (using A, C, G, T/U) that could bind to the molecule with SMILES: {self.target_smiles}.
        Each sequence should be between 20-80 nucleotides long, with GC content around 40-70%, and diverse motifs (avoid long repeats like GGG... or AAA..., aim for balanced nucleotide distribution).
        Aptamers are short DNA or RNA molecules that fold into specific 3D structures to bind targets.

        Examples of good aptamers:
        - GGGAGAATTCCCGCGGCAGAAGCCCACCTGGCTTTGAACTCTATGTTATTGGGTGGGGGAAACTTAAGAAAACTACCACCCTTCAACATTACCGCCCTTCAGCCTGCCAGCGCCCTGCAGCCCGGGAAGCTT
        - GGGAAGGGAAGAAACUGCGGCUUCGGCCGGCUUCCC

        Output only the sequences, one per line, no extra text.
        """
        elif self.prompt_version == 3:
            prompt = f"""
        Generate {num_sequences} unique DNA or RNA aptamer sequences (using A, C, G, T/U) that could bind to the molecule with SMILES: {self.target_smiles}.
        Each sequence should be between 20-80 nucleotides long, with GC content around 40-60%, high entropy (diverse nucleotides, avoid repeats longer than 3), and balanced A/C/G/T distribution.
        Aptamers are short DNA or RNA molecules that fold into specific 3D structures to bind targets. Prioritize sequences similar to real aptamers from datasets like AptaBench.

        Examples of good aptamers:
        - GGGAGAATTCCCGCGGCAGAAGCCCACCTGGCTTTGAACTCTATGTTATTGGGTGGGGGAAACTTAAGAAAACTACCACCCTTCAACATTACCGCCCTTCAGCCTGCCAGCGCCCTGCAGCCCGGGAAGCTT
        - GGGAAGGGAAGAAACUGCGGCUUCGGCCGGCUUCCC

        Output only the sequences, one per line, no extra text.
        """
        else:
            raise ValueError("Invalid prompt_version. Use 1, 2 or 3.")

        if self.examples:
            prompt += "\nPreviously successful examples:\n" + "\n".join(self.examples[:5])  # Топ 5

        response = self.client.models.generate_content(
            model='gemini-flash-latest',
            contents=prompt
        )
        sequences = [line.strip() for line in response.text.split('\n') if line.strip() and len(line.strip()) > 10]
        
        # Фильтрация для prompt_version 3
        if self.prompt_version == 3:
            sequences = [seq for seq in sequences if 20 <= len(seq) <= 80 and 0.4 <= self.gc_content(seq) <= 0.6 and self.shannon_entropy(seq) > 3 and self.longest_homopolymer(seq) <= 3]
        
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
    api_key = "AIzaSyBGZi2TGNeBAgUswjr-sq3TI9RfH-0y4SI"
    generator = LLMGenerator(api_key, prompt_version=1)  # Или 2 для нового prompt
    best = generator.generate_and_evaluate(num_sequences=5, iterations=2)
    print("Best sequences:", best)