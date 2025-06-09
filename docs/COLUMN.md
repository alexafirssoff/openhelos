# Column

A `Column` is a hierarchical processing unit that orchestrates a layered stack of *simplicial neurons* (see [SIMPLICIAL_NEURON.md](./SIMPLICIAL_NEURON.md)) to model the structure and regularities of sequential data. It is designed to serve as a *modular expert* that self-organises to specialise in particular input patterns (e.g., an alphabet or a class of symbols), driven by exposure to data and adaptive modulation signals.

## Purpose

The principal aim of a Column is to learn distributed, context-sensitive representations of input sequences in a fully unsupervised, lifelong manner. Each column processes fixed-width windows of input data, activating a cascade of neurons arranged in layers. Through repeated exposure and competition within a pool, individual columns become *experts* for certain domains (e.g., Cyrillic, Latin, Greek), as evidenced by their familiarity responses.

## High-level Structure

* **Layers:** Each column is organised into `input_width - 1` layers. Each layer consists of a set of simplicial neurons, with the number of neurons per layer increasing towards the bottom.
* **Neurons:** Each neuron in a given layer processes a pair of elements from the previous layer, encoding pairwise (and higher-order) structural relationships in the data.
* **Vertex Map:** The column maintains a mapping of unique observed symbols to vertex identifiers, ensuring a consistent structural representation.

## Parameters

| Parameter                    | Type                           | Description                                                                                                        |
| ---------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `name`                       | `int`                          | Unique integer identifier for the column.                                                                          |
| `input_width`                | `int`                          | The width of the input window (i.e., number of elements processed per step); must be ≥2.                           |
| `storage_path`               | `Path`                         | Path to the parent directory where the column's data and metadata will be persisted.                               |
| `max_overall_fs_history_len` | `int` (default: 30)            | Length of history retained for overall familiarity score (`overall_fs_history`) to assess learning stability.      |
| `overall_fs_history`         | `deque[float]`                 | Rolling history of average familiarity scores over training steps, for monitoring convergence and stability.       |
| `training_steps_count`       | `int`                          | Counter of the number of training updates applied to this column.                                                  |
| `global_vertex_map`          | `dict[Any, int]`               | Mapping from observed symbols (e.g., characters) to global vertex IDs within this column.                          |
| `next_vertex_id`             | `int`                          | The next available unique ID for new input symbols encountered.                                                    |
| `neurons`                    | `List[List[SimplicialNeuron]]` | The main processing layers, each a list of neurons; layer 0 is the top (fewest neurons), last layer is the bottom. |
| `meta_storage`               | `BinaryStorage`                | Manages loading and saving of all metadata for the column, including training history and vertex maps.             |

## Key Methods

* `__init__(...)`: Constructs and initialises a column, optionally loading from persistent storage.
* `_process_data(data, modulation_signal)`: Performs an ascending pass, activating neurons and collecting local metrics (free energy, stability).
* `predict(data)`: Evaluates the familiarity and stability of each element in a window without training (inference mode).
* `train(data, modulation_signal=1.0)`: Updates the internal state of the column (and all contained neurons) using the provided input and modulation signal.
* `dump()`: Serialises and saves the state of the column and all contained neurons.

## Rationale & Function

* **Lifelong and Unsupervised:** Columns learn in a continual, unsupervised fashion, without explicit labels or boundaries. Each column's exposure and internal initialisation biases (e.g., random weights) promote diversification and specialisation.
* **Emergent Expertise:** Through competition and adaptive modulation, some columns become "experts" for specific symbol classes (e.g., alphabets), with others specialising in different domains or contexts.
* **Structural Sensitivity:** By maintaining a vertex map and layering neurons that compute structural metrics (stability, free energy), each column models not just symbol frequency but their combinatorial and topological relationships.
* **Modulation and Stability:** The column tracks the overall familiarity (FS) and uses rolling history to monitor its learning progress and adapt its training intensity.

## Hyperparameters

* `input_width`: Determines the window size and hence the granularity of local structure the column can capture.
* `max_overall_fs_history_len`: Controls the span of historical stability/familiarity scores kept for convergence monitoring.
* `K_FAMILIARITY`: Used in the conversion of free energy to familiarity score (see `calculate_familiarity_score`).
* Others are inherited from contained Simplicial Neurons.

## Usage

A typical workflow involves creating a pool of Columns with different initial seeds and exposing each to different data streams. Over time, each column self-organises into a specialist for a particular structural domain, forming the basis for distributed, emergent symbol representations within the system.

---

## Reference

If you use or extend this code, please cite:

* Firssoff, A. (2025). *HELOS: Hierarchical Emergence of Latent Ontological Structure*.
  [https://zenodo.org/records/15592833](https://zenodo.org/records/15592833) /
  [https://github.com/alexafirssoff/openhelos](https://github.com/alexafirssoff/openhelos)


---

*This is an experimental module; further improvements and optimisations are in progress. See the repo for the latest updates and more advanced approaches.*

---