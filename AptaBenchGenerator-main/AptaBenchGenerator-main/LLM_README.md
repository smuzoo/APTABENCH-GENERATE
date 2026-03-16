# LLM-Enhanced Aptamer Generation with AptaBench

## Overview
This project integrates Google's Gemini AI (via API) with the AptaBench predictive model to generate and evaluate DNA/RNA aptamer sequences. The system uses iterative generation with few-shot learning to improve aptamer quality over time.

## What Was Done
- **API Integration**: Connected to Gemini API using the `google-genai` package.
- **Aptamer Generation**: LLM generates sequences based on a prompt describing aptamers and target molecule (glyphosate by default).
- **Evaluation**: Generated sequences are scored using the AptaBench LightGBM model for binding probability.
- **Self-Learning**: Top-performing sequences are added to the prompt as examples for subsequent generations, enabling iterative improvement.

## How It Works
1. **Initialization**: `LLMGenerator` class initializes with API key and target SMILES.
2. **Generation**: Sends a prompt to Gemini to create aptamer sequences.
3. **Evaluation**: Uses `AptamerLigandPredictor` to compute binding probabilities.
4. **Iteration**: Sorts results, adds top sequences to examples, and repeats.

## Where the Prompt Is Written
The prompt is defined in the `generate_sequences` method of `LLMGenerator` (file: `aptamer_model/src/llm_generator.py`):

```python
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
    prompt += "\nPreviously successful examples:\n" + "\n".join(self.examples[:5])
```

## What to Run
1. Ensure dependencies are installed (`pip install google-genai` and others from `requirements.txt`).
2. Run from the project root:
   ```python
   import sys
   sys.path.append('aptamer_model')
   from src.llm_generator import LLMGenerator

   api_key = "YOUR_API_KEY"
   generator = LLMGenerator(api_key)
   best = generator.generate_and_evaluate(num_sequences=10, iterations=5)
   print("Best aptamers:", best)
   ```

## Results Obtained
- **Test Run**: Generated 2 sequences with binding probabilities ~0.66-0.72 (e.g., 'GGGAGAAUUCGACCCUGAUGAUCCCGUCGCUCCAUUCGACGUGGUUCCUUCGAGAU' with 0.717).
- **Output**: List of sequences with probabilities >0.5 from top results.

## How to Get New Results
- Increase `num_sequences` (e.g., 100) for more candidates.
- Run more `iterations` (e.g., 10) for better learning.
- Change `target_smiles` to another molecule.

## How to Improve
- **More Iterations**: Higher `iterations` refines examples.
- **Better Prompts**: Add diversity instructions or constraints.
- **Filtering**: Only add sequences with prob > 0.8 to examples.
- **RL Integration**: Implement reinforcement learning (e.g., using scores as rewards for fine-tuning).

## How to Further Improve the Prompt
- **Add Context**: Include more biological details (e.g., "Aptamers often have GC-rich regions for stability").
- **Chain-of-Thought**: Ask LLM to reason step-by-step: "First, consider binding motifs, then generate."
- **Examples from Data**: Pull real examples from `AptaBench_dataset_v2.csv` with high pKd.
- **Temperature/Sampling**: If API allows, adjust creativity (higher for diversity).
- **Multi-Turn**: Use conversation history for refinement.
- **Validation**: Add checks for sequence validity (e.g., no invalid bases).

For advanced RL, consider integrating with libraries like Stable Baselines or custom fine-tuning on Gemini.

---

## Notes on model behavior

### 1) Evaluator is “optimistic” on random sequences
When running the predictor on completely random A/C/G/T sequences, the LightGBM model
still returns **high probabilities** (often >0.8 or >0.9). This means the model is biased
in favor of many inputs and does not behave like a crisp "good vs bad" classifier.

### 2) What tends to get a low score (label=0)
In practice, the model gives low probability to sequences that are:
- very long (>>50) and/or
- contain long runs of the same base (e.g., long stretches of `GGG...` or `AAA...`) and/or
- are highly repetitive (repeating motifs)

### 3) How to interpret outputs
Because the model is not calibrated as a true probability estimator, it is safest to:
- treat `proba` as a **relative score**, not an absolute confidence
- use a high threshold (e.g., 0.8 or 0.9) if you want to select "strong" candidates
- compare generated sequences to known positives (e.g., from `AptaBench_dataset_v2.csv`) if possible