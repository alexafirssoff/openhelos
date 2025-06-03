# Emergent Morphemiser via Free Energy Minimisation
⚠️ This is experimental code accompanying a research publication. It is not yet optimised for production or large-scale usage. Use at your own risk. Feedback and contributions are welcome.

## Overview

This module implements an emergent morpheme discovery system guided by a variational Free Energy Principle (FEP). It recursively infers hierarchical binary structures over character sequences, leveraging surprise and prior-based complexity to find optimal segmentations. The system is adaptive, probabilistic, and data-driven, yet maintains interpretability through clear hierarchical structures.

## Key Components

### 1. `Node`

Immutable representation of a parse tree node:

* **Leaf:** a single grapheme (e.g., 'а')
* **Internal:** a binary pair of nodes

Provides compact representation and leaf extraction.

### 2. `PredictiveModel`

Lightweight context-based predictive model estimating:

* **Surprise = -log P(next | context)** with Laplace smoothing
* Frequency-based update rule
* Used for both next-token and internal-node prediction

### 3. `node_registry`

Centralised memory for all known nodes:

* Stores log-priors, predictors, and usage stats
* Enables complexity calculation and adaptive learning

### 4. `SearchState`

Encapsulates a hypothesis during parsing:

* Free Energy (complexity + surprise)
* Root node and span info
* Supports priority-based beam search

### 5. `find_best_parses_fep`

CKY-style beam search algorithm:

* Parses all spans in bottom-up manner
* At each step, calculates FE = -log P + PE + structural penalty
* Keeps top hypotheses per span

### 6. `process_word_fep_v3`

Main entry point:

* Accepts a word, returns top-K parses
* Optionally updates priors and predictors
* Visualises best parse using Graphviz (optional)

## Free Energy Design

For each hypothesis:

```
Free Energy = Tree Complexity + Predictive Surprise + Composition Penalty
```

* **Tree Complexity:** sum of -log P(node) from registry
* **Predictive Surprise:** mean prediction error from:

  * Parent predicting children
  * Left child predicting right sibling
* **Penalty:** small constant cost per composition step

## Learning

Upon each successful parse:

* Priors are updated to reflect reuse and predictive value
* Predictive models updated with observation counts
* Nodes used in good explanations are favoured

## Visualisation

Optional visualisation via `graphviz`:

* Shows internal structure of the parse
* Nodes annotated with frequency and log-prior

## Output

Console output for each parsed word includes:

* Best parses with FE breakdown
* Tree string representation
* Optional PNG of tree structure

## Use Case

This module can be used to:

* Discover morpheme-like substructures in raw text
* Support unsupervised morphology learning
* Perform interpretable structural parsing for cognitive or linguistic modelling

## Integration

Can be embedded into larger FEP-based systems, AGI research, or linguistic pipelines. All learning is incremental, making it suitable for continual learning from a text stream.

## Requirements

* Python 3.8+
* `graphviz` (with `dot` installed)
* `dataclasses`, `heapq`, `math`, `collections`

## Licence

This module is published under the **NonCommercial-Research** licence. See `LICENSE.txt` for details.

## Author

Alexei Firssoff
ORCID: [0009-0006-0316-116X](https://orcid.org/0009-0006-0316-116X)

---

For documentation or visualisation suggestions, open an issue or reach out!
