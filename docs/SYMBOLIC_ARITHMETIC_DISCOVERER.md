# HELOS–Topos Arithmetic Learner

> **From labelled examples to emergent algorithms — without hard‑coded arithmetic.**

This project demonstrates a compact, interpretable learner that acquires the digit‑wise algorithms for **addition**, **subtraction (absolute difference)**, and **multiplication** directly from **labelled training triples** `(a, op, b → c)`. At inference time the system performs branch‑free, per‑digit reasoning over a learned **sheaf of local rules**. There is **no embedded arithmetic logic**: no manual carries, no sign rules, no multiplication tables. Instead, the model discovers and composes the rules needed to reproduce schoolbook arithmetic.

---

## Why this matters

Most symbolic learners either memorise examples or hide the target algorithm in priors. Here, we:

* **Avoid a priori arithmetic**: the code does not encode the semantics of `+`, `−`, or `×` during inference. The few necessary helpers (e.g. order hints) are **learned** from data.
* **Expose the learned rules**: the global section of each sheaf is an explicit mapping from local contexts to outputs (digit and next state). You can dump, read, and audit it.
* **Generalise out of distribution (OOD)**: we evaluate on longer numbers, skewed digit distributions, and true multi‑digit multiplication that was **not** seen as such during training. Strong OOD accuracy indicates the model **abstracted the algorithm**, rather than memorising whole numbers.

---

## How it works

* **Topos of local rules**: The base space consists of per‑digit contexts `(op, d1, d2_or_m, carry_in)`. Each point has a **stalk** of possible values `(digit_out, carry_out)`. A **global section** selects one value per point so that local predictions compose into a consistent number.
* **HELOS perspective**: We treat inference as selecting a minimal‑tension global section that best explains observations. Conceptually, the model reduces **free energy** by choosing rules that make digit transitions predictable across the number.
* **Learning by EM**: We maintain a small mixture of sheaves. Observations (labelled triples) are decomposed into local steps. The E‑step assigns responsibility; the M‑step updates per‑point counts. A smoothed MAP then yields the global section.
* **No hidden arithmetic branches**: Subtraction order (which operand is larger), the useful order for addition digits, and a fallback projection for multiplication are all **learned auxiliaries**, not hand‑crafted rules.

---

## What counts as no cheating

* **No harcoded entities to force addition logic** at inference; the same lookup mechanism handles all operators.
* **No integer comparisons** during inference for subtraction. We evaluate **both operand orders** using learned rules and pick the one with higher learned consistency score.
* **No carry formulae** are coded; carries are just part of the learned stalk values.

Training does use a **labelled dataset** (like teaching a child with marked exercises) and a **training‑only decomposer** to extract clean local steps as supervision. Inference never sees those teacher signals.

---

## OOD evaluation (what and why)

We run three stress tests to show the learner derived the *procedure*:

1. **Length↑** – numbers longer than in training. If accuracy holds, the model composes local rules linearly in length.
2. **Digit‑skew** – inputs biased towards rare digits (e.g. 7s, 9s). Robustness indicates the model is not tied to training frequencies.
3. **Multi‑digit `×`** – full long multiplication via learned single‑digit `×` plus learned addition of shifted partials. Success here shows **algorithmic reuse**: the model performs proper long multiplication without ever seeing it as a monolithic operator during training.

Consistently high OOD accuracy strengthens the claim that the system **internalised the algorithm**, not merely stored seen patterns.

---

## Results (excerpted)

```
Total Train: 10500, Total Test: 2700

Starting Training (HELOS Topos)…
… (50 emergent sheaves)
Epoch 6/6  Time: 1.41s  Sheaves: 50 | Add:0.997  Sub:0.993  Mul:0.991 | Tension[unknown:5, mism:6]

— Random Test Summary —
Accuracy +: 0.9936   (155/156)
Accuracy -: 0.9758   (161/165)
Accuracy *: 0.9609   (172/179)
Overall:  0.9760   (488/500)

— OOD Evaluations —
OOD length↑:     acc = 0.9833   (295/300)
OOD digit-skew:  acc = 0.9900   (297/300)
OOD multi-digit ×: acc = 0.9600 (288/300)
```

### Traceable reasoning (sample)

```
831 - 178 = 653 (True: 653) ✓
[−] sign decision feats=('-', 3, 3, '8', '1', 0, '1', '8') → flip_hint=0 | chosen='hint'
[-] i=0 s_in=0 key=('-', '1','8',0) → d_out=3, s_out=1
[-] i=1 s_in=1 key=('-', '3','7',1) → d_out=5, s_out=1
[-] i=2 s_in=1 key=('-', '8','1',1) → d_out=6, s_out=0
```

```
1234 × 56 = 69104 (True: 69104) ✓
[×] partial for m=6 @pos=0 → 7404
[+] accumulate 0 + 7404 → 7404
[×] partial for m=5 @pos=1 → 6170
[×] shift by 1 → 61700
[+] accumulate 7404 + 61700 → 69104
```

---

## Summary table

| Split / Metric                  | `+` acc | `−` acc | `×` acc | Overall |
| ------------------------------- | :-----: | :-----: | :-----: | :-----: |
| In‑distribution test            |  0.9936 |  0.9758 |  0.9609 |  0.9760 |
| OOD: length↑                    |    –    |    –    |    –    |  0.9833 |
| OOD: digit‑skew                 |    –    |    –    |    –    |  0.9900 |
| OOD: multi‑digit `×` (composed) |    –    |    –    |  0.9600 | 0.9600* |

* Overall shown for the dedicated multi‑digit `×` OOD set.

---

## Quick start

```bash
# 1) Install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# 2) Run
python src/core/symbolic_arithmetic_discoverer_v1.py

# 3) Inspect outputs
#   • results/arithmetic/helos_results.csv        – random test suite
#   • results/arithmetic/helos_traces.txt         – human‑readable traces
#   • results/arithmetic/helos_memory_dump.json   – learned rules (global sections)
```

# Running the Arithmetic Discovery Experiment

This section explains how to execute and configure the **HELOS–Topos Arithmetic Learner** experiment.

---

## Basic command

```python
run_experiment(seed=21, epochs=6, train_n=3500, test_n=900, max_len=3)
```

This function launches a complete training–evaluation cycle, constructing the sheaf-based arithmetic learner, running optimisation, and performing both in-distribution and OOD tests.

---

## Parameter overview

| Parameter     | Type  | Default | Description                                                                                                                                                                                                            |
| ------------- | ----- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`seed`**    | `int` | `21`    | Controls the random generator for data sampling and initialisation. Set this to ensure reproducible results. Different seeds change the stochastic decomposition and initial rule assignment.                          |
| **`epochs`**  | `int` | `6`     | Number of full passes over the training data. Each epoch refines sheaf specialisation and reduces internal 'tension' between local and global rules. More epochs typically improve stability but may slow convergence. |
| **`train_n`** | `int` | `3500`  | Number of labelled expressions `(a, op, b → c)` used for training. Increasing this improves the model’s exposure to rare digit combinations and leads to stronger OOD generalisation.                                  |
| **`test_n`**  | `int` | `900`   | Number of test expressions used for the main evaluation. This does not affect learning but provides accuracy statistics and random expression traces.                                                                  |
| **`max_len`** | `int` | `3`     | Maximum number of digits per operand during training. Acts as an upper bound on local compositional contexts. Lower values accelerate training; higher values test scaling and algorithmic compositionality.           |

---

## Example usage

```bash
# Default quick run
python -m src.core.symbolic_arithmetic_discoverer_v1

# Custom experiment with longer numbers and larger dataset
python -c "from src.core.symbolic_arithmetic_discoverer_v1 import run_experiment; run_experiment(seed=7, epochs=8, train_n=10000, test_n=2500, max_len=4)"
```

### Notes

* Larger `max_len` increases difficulty exponentially but provides stronger evidence of learned generalisation.
* You can inspect logs, sheaf structures, and result traces in `results/arithmetic/` after each run.
* The learner is deterministic per seed and configuration — ideal for controlled ablation or replication studies.

---

**Tip:** to perform a reproducibility test, run the same configuration with two different seeds and compare the resulting sheaf topologies in the memory dump JSON.


---

## Implementation notes

* **Training**: labelled triples only; local steps are provided by a training‑only decomposer to stabilise learning of stalk transitions.
* **Inference**: purely via learned global sections and three tiny auxiliaries (sign hint for `−`, learned digit‑order for `+`, star‑projection fallback for `×`). No integer maths or carry rules are hard‑coded.
* **Complexity**: inference cost is linear in the number of digits; memory is the size of the learned sections.

---

## Citing or discussing

If you reuse this code or the learning pattern, please reference this repository and call out the **sheaf‑based, prior‑free inference** and **OOD composition tests** as the key novelty.
