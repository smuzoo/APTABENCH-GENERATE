# AptaBenchGenerator — Glyphosate Aptamer Generation

## Scope

Target molecule: **Glyphosate**  
Dataset and scoring models: **AptaBench**  

Goal: generate and benchmark aptamer candidates using multiple strategies and select top candidates for lab validation.

## Repository Rules

`aptamer_model/`
- Contains the trained AptaBench predictive model.
- Must be used for scoring all generated sequences.
- Do not modify or duplicate this component.

### Generation Development
- Each strategy must be implemented in a **separate folder**.
- Development must occur in a **separate Git branch**.
- All scoring must call the shared model from `aptamer_model/`.

Each strategy must:
1. Generate 100 sequences
2. Score them using the shared model
3. Output ranked candidates
---

## Generation Strategies

- Genetic Algorithm (GA)
- Reinforcement Learning (RL)
- LLM fine-tuning

---

## Evaluation Criteria

### Sanity Checks
- % unique sequences
- Mean Levenshtein distance within set
- Closeness to AptaBench distributions (GC, length, k-mers)
- Samples per hour

### Validation
- Success rate (random split model)
- Success rate (molecule-disjoint model)
- % with mean ipTM > 0.7 (AF3, averaged over 5 runs)
- % with mean pLDDT > 70 (AF3, averaged over 5 runs)

---

## Recent Experiments: Glyphosate Model Retraining

### What We Did
- Combined the original AptaBench dataset (v2) with glyphosate-specific interaction data (64 examples from experimental records).
- Retrained the LightGBM predictive model on the merged dataset.
- Evaluated the retrained model on previously generated aptamer candidates, random sequences, and counter-target sequences (e.g., for AMPA, glufosinate).

### Why We Did It
- The original AptaBench model was trained on general aptamer-target interactions, not glyphosate-specific.
- Goal: improve the model's ability to score glyphosate-binding aptamers more accurately and distinguish true positives from negatives/random sequences.

### Results
- Retraining completed successfully: the new model achieved a train ROC-AUC of 0.9999 on the combined dataset.
- However, the model shows signs of overfitting: it assigns very high probabilities (~0.99-1.0) to all evaluated sequences, including random and counter-targets.
- Comparison with old model: new model gives an average +0.08 higher probability scores on the same generated candidates.

### Why It Didn't Work as Expected
- Small addition of glyphosate data (only 64 examples), insufficient for significant improvement.
- No class balancing (positives vs. negatives) or holdout validation set used during retraining.
- Lack of strong negative examples in the training data, leading to poor generalization.

### Next Steps
- Use the original AptaBench model as the baseline for scoring.
- Focus on generation strategies combined with post-hoc filters (e.g., GC content, sequence length, Shannon entropy).
- If retraining is attempted again, incorporate more diverse negatives, class balancing, and cross-validation.

## Experimental Phase

Top 10 candidates per strategy → laboratory validation.

---

## Deadlines

- Strategy plan (pseudo-code) — Mar 1  
- First checkpoint — Mar 8  
- Final strategy implementation — Mar 16  
- Generated set benchmarking — Mar 20  
- AF3 benchmarking — Mar 26  
- Lab validation — Apr 10  

