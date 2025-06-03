# Lifelong Symbol Clusterer

⚠️ This is experimental code accompanying a research publication. It is not yet optimised for production or large-scale usage. Use at your own risk. Feedback and contributions are welcome.

A Python implementation of an **emergent clustering system** for symbolic data, grounded in the **Free Energy Principle (FEP)** interpreted through topological stability of co-occurrence patterns. This system allows symbols to self-organise into stable, reusable structures (clusters), which then participate in further segmentation of input data.

## ✨ Key Concepts

### Free Energy in Discrete Spaces

This clusterer applies the **FEP** in a novel domain: **discrete symbolic streams** such as text. Rather than relying on continuous prediction error, this system measures **topological variation** of co-occurrence graphs over time, using **Persistent Homology**. Clusters are created only when this variation minimises — i.e. when the structure becomes *epistemically stable*.

### Persistent Homology as Predictive Invariance

Topological persistence diagrams reflect how connected components and holes in the symbol graph appear and disappear across filtration scales. We interpret the stability of these diagrams across time as an analogue to low free energy — the structure has become predictable and thus symbolisable.

### Lifelong and Self-Referential Learning

Clusters, once stable, are assigned unique identifiers and **reinserted** into the symbol stream. This allows the system to build **hierarchical, lifelong representations**, continually folding newly observed patterns into an evolving ontology of symbolic categories.

---

## Features

* **Discretely grounded FEP**: No continuous latent space required.
* **Persistent homology** via Gudhi.
* **Self-organising symbolic units** (clusters become symbols).
* **Minimal assumptions**: Only symbol co-occurrence counts are tracked.
* **Stable cluster detection**: Using bottleneck distance and component variance heuristics.
* **Visualisable** state (see examples).

---

## Theoretical Motivation

### Predictive Coding vs Topological Stability

Most FEP-based models rely on variational inference and gradient flows. Here, we substitute:

* **Error minimisation** → **diagram stabilisation**
* **Surprise** → **topological change**
* **Posterior update** → **symbolisation of clusters**

This approach allows applying predictive-coding ideas to purely discrete symbolic environments such as language, genomes, or sequences of actions.

---

## How It Works

1. **Input**: Strings of characters (or any symbolic stream).
2. **Co-occurrence graph** is built over time.
3. **Persistent homology** is computed on the largest active subgraph.
4. If the **topology stabilises** (based on bottleneck distance & point-count variance):

   * Cluster is **symbolised** into a new node.
   * System continues learning, now including the new cluster as a unit.

---

## Visualisation

We provide a companion visualiser that projects the resulting cluster definitions and graph interactions into a 2D space. Each cluster is shown as a colour-coded shape containing its member symbols.

---

## Dependencies

* `gudhi` (Persistent Homology library)
* `numpy`
* `logging`

Install with:

```bash
pip install gudhi numpy
```

---

## Potential Applications

* Cognitive models of reading or perception
* Emergent linguistics (unsupervised morphology)
* Unsupervised parsing
* Neuro-Symbolic AI systems

---

## Licence

This module is published under the **NonCommercial-Research** licence. See `LICENSE.txt` for details.

## Author

Alexei Firssoff
ORCID: [0009-0006-0316-116X](https://orcid.org/0009-0006-0316-116X)

---

## Future Work

* Integration with categorical/topos-theoretic semantics
* Formal FEP grounding through category theory
* Symbolic compression based on cluster reuse
* Support for dynamic segmentation of continuous text streams
