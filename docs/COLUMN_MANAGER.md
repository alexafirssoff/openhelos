# ColumnManager: Neuro-inspired Modular Pool for Emergent Structure Learning

## Rationale

The `ColumnManager` module is inspired by the concept of cortical columns in the neocortex — modular, semi-autonomous processing units, each capable of learning, specialising, and adapting to distinct statistical patterns in the input. The primary aim is to enable *emergent* and *lifelong* discovery of latent structure, segmentation, and context-sensitive “symbols” in sequences without external supervision. The architecture leverages competition, adaptive plasticity, and topological signal processing to allow different columns to specialise, co-exist, and cooperate in the recognition of complex patterns — including alphabets, words, or sub-lexical units.

A key feature of this implementation is *intentional heterogeneity* at initialisation: each column in the pool is randomly initialised with slightly different parameters, ensuring diversity of initial responses and allowing the system to robustly explore and “lock in” different input structures.

---

## Overview

* **ColumnManager** orchestrates a pool of column modules (see [COLUMN.md](./COLUMN.md)), each acting as an independent structure learner and “expert”.
* The manager dynamically assigns training resources, modulates plasticity, and tracks performance using *SumIdealness* and *Modulation Signal* for each column.
* It facilitates adaptive column creation, local retraining, and automatic segmentation of the input.
* The system is designed for lifelong, online operation, continually adapting to changing statistics, context shifts, and novelty.

---

## Initialisation and Heterogeneity

When a new `ColumnManager` is created:

* A pool of `columns_quantity` columns is initialised, each with unique random parameters.
* This *parametric diversity* is crucial: it increases the probability that, as data streams in, at least some columns will initially respond better to certain structures or “alphabets”, quickly becoming *experts* for specific input types. Others may become dormant or specialise later as new data types emerge.

---

## Key Hyperparameters

Defined in `ColumnManagerHyperParams`:

* **vigilance**: Baseline threshold for confidence. Used to determine if a column’s response is “stable” and can be considered an expert for some pattern.

* **prediction\_history\_size**: Number of recent predictions to retain for analysis of each column’s performance.

* **column\_fs\_history\_len**: Length of the rolling history of familiarity scores (FS) within each column, used to assess stability and trends.

* **stability\_trend\_window\_k**, **stability\_slope\_threshold**: Parameters controlling how the manager assesses whether a column has reached stable specialisation.

* **learning\_context\_k**: Determines the width of context considered when a new column is created and trained.

* **min\_segment\_len\_new**, **min\_segment\_len\_train**: Minimum contiguous segment lengths required to trigger new column creation or retraining.

* **std\_dev\_factor\_for\_mask**, **min\_fs\_for\_mask**: Control the adaptive threshold for masking out well-recognised parts of the input, focusing learning on novel or ambiguous regions.

* **dissonance\_history\_len**, **dissonance\_factor**, **min\_idealness\_threshold**: Control detection of “dissonance” (abnormally high uncertainty or change), triggering adaptive mechanisms.

* **min\_overall\_fs\_for\_unstable\_protection**: Prevents training of unstable columns on poorly recognised data.

* **softmax\_temp**, **softmax\_temp\_modulation**, **softmax\_temp\_relative**: Temperatures for softmax operations that convert performance scores into probabilistic modulation signals.

* **threshold\_novelty**, **novelty\_detection\_factor**, **novelty\_history\_len**: Parameters for tracking and responding to novelty, allowing the system to shift plasticity to new experts as data distributions change.

* **common\_element\_threshold**, **common\_set\_min\_columns**: Parameters for identifying elements recognised with high confidence by multiple columns, supporting discovery of common “symbols”.

* **top\_down\_boost\_factor**: Controls the influence of top-down (predictive) signals in modulating column learning.

---

## Core Methods and Dynamics

* **process(window: List\[Any])**: Main entry point. Assigns each input element to the best-matching column(s), modulates plasticity, coordinates training, and returns a stabilised output with symbol IDs.
* **\_calculate\_idealness\_score**: Computes the “idealness” of an element for a column, based on familiarity and contextual surprise.
* **plot\_dynamics**: Visualises the evolving dynamics of SumIdealness and Modulation Signal for each column, allowing inspection of specialisation and competition.

---

## Why This Architecture?

* **Modularity**: Enables distributed, parallel learning — each column can specialise independently, preventing catastrophic forgetting and supporting continual adaptation.
* **Competition and Diversity**: Initial heterogeneity and dynamic modulation ensure that the system can robustly cover diverse input structures.
* **Lifelong Learning**: Supports continuous discovery, specialisation, and memory retention over long time scales, even as input distributions shift.
* **Emergent Symbol Formation**: Rather than predefining symbols, the system lets structure “bubble up” from the statistics and topology of the input, reminiscent of how biological systems might form *proto-symbols* and context-aware units.

---

## Example Usage

```python
from column_manager import ColumnManagerEmergent
from pathlib import Path

c_manager = ColumnManagerEmergent(
    name=0,
    storage_path=Path('./data/c_managers'),
    columns_quantity=8,
    input_width=24
)

window = list('example input string')
winner_id, output = c_manager.process(window)
print(winner_id, output)
```

---

## Reference

If you use or extend this code, please cite:

* Firssoff, A. (2025). *HELOS: Hierarchical Emergence of Latent Ontological Structure*.
  [https://zenodo.org/records/15592833](https://zenodo.org/records/15592833) /
  [https://github.com/alexafirssoff/openhelos](https://github.com/alexafirssoff/openhelos)


---

*This is an experimental module; further improvements and optimisations are in progress. See the repo for the latest updates and more advanced approaches.*

---