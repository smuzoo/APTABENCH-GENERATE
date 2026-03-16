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
