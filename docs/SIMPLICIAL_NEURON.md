# SimplicialNeuron

A **simplicial neuron** is a novel computational primitive designed for *topological, structure-sensitive learning* at the level of individual neurons.
Unlike classical artificial neurons, which operate on scalar activations or simple vector spaces, a simplicial neuron maintains and adapts an **internal simplicial complex** — a combinatorial model of its local experience and input patterns.

This concept was developed to address limitations of purely synaptic memory, hypothesising that real-world cognition requires richer, *structurally grounded* internal representations.

---

## What does the SimplicialNeuron do?

**Core functions:**

* Receives symbols (e.g. glyphs or characters), one at a time or in pairs.
* Builds and updates an internal *simplicial complex* (`self.complex`) as its local memory.
* Tracks the stability and frequency of observed elements (vertices) and their co-occurrences (edges, higher-order simplices).
* Computes **familiarity** and **surprise** scores for inputs, based on how well they fit its existing structure (using Free Energy minimisation).
* Learns by plastic changes in the internal structure, reinforcing familiar patterns and adapting to novel ones.

**In short:**
*A simplicial neuron learns a structural “map” of the input space it observes, and provides structural metrics for use in higher-level modules (columns, pools, etc).*

---

## Key Concepts

### 1. **Simplicial Complex**

* The neuron's internal memory is a *simplicial complex* Δ: a set of vertices (input elements) and simplices (co-occurrence patterns).
* Vertices = unique observed symbols
* 1-simplices (edges) = observed pairs
* Higher simplices = sets of symbols occurring together
* Complex grows and adapts as new input arrives.

### 2. **Stability Metric (`P_s`)**

* For any element σ, *stability* reflects both its frequency and its structural embeddedness within the neuron's local complex.
* High `P_s` means the element is familiar, central, and predictive in this neuron's history.

### 3. **Familiarity Score (`FS`) & Free Energy (`F`)**

* The neuron evaluates new input (single symbol or pair) based on how “expected” it is:

  * *Low Free Energy* (high familiarity): input fits well within existing structure.
  * *High Free Energy*: input is novel or does not align with learned patterns.

### 4. **Local Structural Plasticity**

* Learning is modelled as *plastic adaptation* of the complex:

  * New vertices/edges added for novel input,
  * Edge weights/frequencies adjusted for familiar pairs,
  * Pruning or decay may be applied to obsolete patterns.

---

## Basic Usage

```python
from simplicial_neuron import SimplicialNeuron

from pathlib import Path
neuron = SimplicialNeuron(0, Path('./data/neurons'), hyperparameters=NeuronHyperParams)
S, P, F, b0, z, el1, el2 = neuron.ascending_activation(ord('a'), ord('b'))
print(f"S={S}, P_pair={P}, F={F}, b0={b0}, z={z}, P_el1={el1}, P_el2={el2}")
```

---

## Hyperparameters
Absolutely! Here is the same **Simplicial Neuron** documentation in clear, technical British English, reflecting the intent and detail of your code and design.

---

# SimplicialNeuron: Documentation

## Overview

The `SimplicialNeuron` is a computational unit that models local analysis of the topological structure of input sequences. Unlike classical neurons, its internal memory is implemented as a **simplicial complex** (i.e., a graph-like structure) which adapts and accumulates structural patterns as edges between observed symbols.

**Key concepts:**

* **Local topological memory:** Graph-based representation of pairwise relationships between symbols.
* **Learning through structural plasticity:** The internal complex grows and changes as new data is observed.
* **Metric evaluations:** For each input, the neuron computes stability, energy, and familiarity scores.

---

## Hyperparameters (NeuronHyperParams)

A complete list of hyperparameters and their function:

| Name                            | Type    | Default     | Description                                                                                                                                    |
| ------------------------------- | ------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `epsilon`                       | `float` | `1e-10`     | A small number to avoid division by zero and log(0) errors.                                                                                    |
| `freq_damping_factor`           | `float` | `10.0`      | Damping coefficient for reducing the influence of frequency in the free energy (F) calculation. Larger values slow the influence of frequency. |
| `alpha_p_single`                | `float` | `0.2`       | Weight for the frequency component in the stability formula (`P_single`). A value of 0.2 means 20% frequency, 80% topological centrality.      |
| `k_familiarity`                 | `float` | `0.7`       | Exponential coefficient used in the transformation from free energy (F) to Familiarity Score (FS): `FS = exp(-k*F)`.                           |
| `max_history_s`                 | `int`   | `1000`      | Maximum history length for metrics (used for averaging, adaptation, and limiting memory).                                                      |
| `default_plasticity_control`    | `str`   | `'S_based'` | Plasticity control mode: `'S_based'` — plasticity depends on stability S, `'F_based'` — on average F, `'none'` — always maximal.               |
| `default_plasticity_decay_rate` | `float` | `5.0`       | Exponential decay parameter for plasticity: larger values mean plasticity decays faster as S or FS increases.                                  |
| `min_modulation_for_history`    | `float` | `0.1`       | Minimum modulation value for updating history and learning weights; allows separation of train/predict phases.                                 |
| `p_single_maturity_start`       | `int`   | `1`         | Minimum number of vertices before “maturity” starts to influence element stability calculation.                                                |
| `p_single_maturity_ramp`        | `float` | `3.0`       | “Maturity ramp” length — after this many vertices, maturity saturates to 1.0.                                                                  |
| `w_init`                        | `float` | `0.1`       | Initial weight for edges when a new connection is formed.                                                                                      |
| `eta_w`                         | `float` | `0.01`      | Edge weight learning rate (Hebbian-style update).                                                                                              |
| `gamma_w`                       | `float` | `0.001`     | Edge weight decay coefficient.                                                                                                                 |
| `lambda_w`                      | `float` | `1.0`       | Coefficient controlling the influence of edge weights in the free energy calculation (if used).                                                |

---

## Methods: Brief Descriptions

* **`__init__`**: Initialises or loads the state of the neuron (simplicial complex, weights, frequencies, history).
* **`_get_local_plasticity_level`**: Determines the neuron's current level of plasticity — from 1.0 (maximum) to 0.0 (locked). Depends on the chosen mode.
* **`_touch_vertex`, `_touch_pair`**: Adds vertices and edges to the complex (graph), updates frequencies and weights, initialises weights with unique random noise per neuron.
* **`_update_edge_weight`**: Updates the edge weight depending on familiarity and free energy metrics.
* **`compute_stability`, `pair_stability`, `single_stability`**: Compute various metrics of structural stability: global S, pairwise (P\_pair), single-element (P\_single) stabilities.
* **`compute_free_energy`**: Computes free energy F for a pair — a measure of structural novelty or surprise.
* **`compute_b0`**: Calculates the number of connected components (Betti-0 number) in the complex.
* **`ascending_activation`**: Main processing step — updates the structure, computes metrics, updates weights, returns all characteristics for the given input pair.
* **`plot_metrics_history`**: Visualises metric histories over time (S, P\_pair, FS, F, b0).
* **`dump`**: Saves the neuron's state to disk.
---

## Main Methods

| Method                   | Purpose                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------- |
| `observe(symbol)`        | Observe a single symbol (vertex), update memory.                                                        |
| `observe_pair(a, b)`     | Observe a pair (edge), update structure, weights, frequencies.                                          |
| `familiarity(symbol)`    | Return the stability/familiarity score for a given symbol.                                              |
| `familiarity_pair(a, b)` | Return the familiarity/free energy score for a pair, given current complex.                             |
| `update_structure()`     | (Optional) Recompute or prune the internal simplicial complex, based on decay and frequency thresholds. |

---

## Visualisation

You can plot the neuron's current simplicial complex (vertices, edges, weights) to interpret what it has “learned” about the data space.

---

## How does this differ from a classic neuron?

* **Classic neuron:** Scalar activation, fixed weights, no explicit structure.
* **Simplicial neuron:** Rich, structural memory; learns patterns as combinatorial topology; outputs metrics reflecting not just frequency but *structural centrality* and contextual expectation.

---

## Rationale

This approach is motivated by the hypothesis that intelligence and memory require *local topological processors*, not just distributed vector codes or connection weights.
A simplicial neuron thus acts as a *local structure detector*, forming the basis for higher-level modular systems (columns, pools, memory hierarchies).

---

## Reference

If you use or extend this code, please cite:

* Firssoff, A. (2025). *HELOS: Hierarchical Emergence of Latent Ontological Structure*.
  [https://zenodo.org/records/15592833](https://zenodo.org/records/15592833) /
  [https://github.com/alexafirssoff/openhelos](https://github.com/alexafirssoff/openhelos)


---

*This is an experimental module; further improvements and optimisations are in progress. See the repo for the latest updates and more advanced approaches.*

---
