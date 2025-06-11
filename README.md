# OpenHELOS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15592833.svg)](https://doi.org/10.5281/zenodo.15592833)

<!--[![arXiv](https://img.shields.io/badge/arXiv-2406.xxxxxv1-B31B1B.svg)](https://arxiv.org/abs/2406.xxxxx)-->

OpenHELOS (**H**ierarchical **E**mergence of **L**atent **O**ntological **S**tructure) is an experimental codebase developed to empirically investigate several phenomena predicted by the theoretical framework proposed in the associated research paper, grounded in the Free Energy Principle:

1. **Lifelong Symbolic Clusterer** — unsupervised emergent clustering using co-occurrence and persistent homology.
2. **Emergent Morphemiser** — a recursive parser that discovers morphological structure via Free Energy minimisation.

It integrates predictive coding, free energy minimisation, and discrete structure composition, aiming at a scalable model of cognitive parsing.

The code constructs the very first cognitive primitives described in the paper — morphemisation, clustering, symbolisation — from energetic and statistical principles.

The aim of the presented experiments is **not to outperform state-of-the-art systems across a wide range of benchmarks**, but rather to demonstrate that the foundational premises articulated in the accompanying paper are, in principle, realisable. In particular, the results highlight that even an approximated application of the Free Energy Principle (FEP) to discrete symbolic structures—such as hierarchical morpheme parsing—can yield promising and coherent outcomes. This serves as an initial validation of the theoretical framework underpinning HELOS, rather than a conclusive empirical comparison.

Training such systems requires carefully constructed, stepwise datasets that differ fundamentally from those typically employed in standard machine learning pipelines. Rather than relying solely on large volumes of labelled data, these systems benefit from inputs that are curated to reflect the internal logic and compositional structure of the target domain. In the case of morphological parsing, this means exposing the model to examples that progressively reveal morphemic patterns in a systematic and hierarchical fashion, thereby enabling the system to internalise morphemes not merely as surface patterns, but as structured units embedded in a broader generative framework.

> 🚨 This is **experimental research code** developed as part of an ongoing effort to build explainable-AI-aligned linguistic self-organisation systems. Not production-ready.
> The current calculation of metrics such as F1 and precision for HELOS **is incorrect** — this is a bug that we will **fix soon**. The calculation for the publication was done **manually**; please refer to the actual morpheme segmentation for guidance.

I’ve also created a brief overview of the article in a more visual format to illustrate how I arrived at the ideas and thought process described in the article. Here’s the link: [the overview on Medium](https://medium.com/@alexfirssoff/from-symbols-to-cognition-my-journey-building-a-cognitive-framework-ec6da99c5a09).

---

## 🔬 Research Intent

This repository is part of a larger research initiative. It demonstrates how complex linguistic structure can **emerge without supervision** under energy-based principles.

---

## 🌐 Project Structure

```plaintext
openhelos/
├── datasets/                      
│   ├── dicts/                    // Dataset for the morpher
│   │   ├── deu/
│   │   ├── fra/
│   │   ├── rus/
│   │   ├── tur/
│   └── fep_ph_clusterer/         // Dataset for the clusterer
├── docs/
├── results/
│   ├── fep_morpher/
│   └── fep_ph_clusterer/
├── temp/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── fep_morpher.py
│   │   ├── fep_ph_clasterer.py
│   └── experimental_setup/
│   │   ├── __init__.py
│   │   ├── clusterer_streamlit.py
│   │   ├── morpher_setup.py
│   │   └── morpher_streamlit.py
│   ├── __init__.py
├── clusterer_test.py             // Entry point for running the experiment with the morpher.
├── morpher_test.py               // Entry point for running the experiment with the clusterer.
├── LICENCE.txt
├── README.md
├── requirements.txt
```

---

## 🧠 Key Concepts

* **Free Energy Principle (FEP):** All parsing and clustering is formulated as the minimisation of free energy = complexity + surprise.
* **Persistent Homology:** Used to detect stable topological features in symbol co-occurrence graphs (clusterer).
* **Recursive Type S:** The morphemiser parses using a binary structure that recursively composes graphemes.
* **Emergence over Supervision:** No labelled data, no morphology dictionaries — all structure is learned from raw text.

---

## 🚀 Getting Started

### 1. Install dependencies

We recommend using a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** You’ll need [Graphviz](https://graphviz.org/download/) and the [Gudhi library](https://gudhi.inria.fr/).

### 2. Run morpher

```bash
streamlit run morpher_test.py
```
Then press 'Run training & comparison'.
This will execute a predefined setup for both the clusterer and morphemiser. 
See `experiment_setup/` and `datasets/` to modify inputs and parameters.

🚨 <span style="color:red">The current calculation of metrics presented in .csv files such as F1 and precision for HELOS is incorrect — this is a bug that we will fix soon. The calculation for the publication was done manually; please refer to the actual morpheme segmentation for guidance.</span>

### 3. Run clusterer

```bash
streamlit run clusterer_test.py
```
Then press 'Generate Clusterer State from Sample Data'.

---

## 📄 Example Datasets

* `datasets/fep_ph_clasterer/plain_strings.txt` — text strings for cluster formation.
* `datasets/dicts/` — optional dictionaries for morpheme emergence evaluation.

---

## 🧪 Features

### FEP-PH Clusterer

* Lifelong tracking of co-occurrences
* Stable cluster detection via persistent diagrams
* Symbolisation of new abstract units

### Morphemiser

* CKY-like beam search parser
* FE-based scoring of hypotheses
* Structural generalisation from raw strings
* Graphviz visualisation support

---

## 🗄 Example Output

See `results/fep_morpher/` and `results/fep_ph_classifier/` for tree graphs, cluster traces and FE scores.

---

## 📚 Documentation

| Component   | Docs                                |
| ----------- | ----------------------------------- |
| Clusterer   | [`CLASTERER.md`](docs/CLUSTERER.md) |
| Morphemiser | [`MORPHER.md`](docs/MORPHER.md)     |

---

## 📄 Licence

See [`LICENCE.txt`](LICENCE.txt)

---

## 🧽 Author

Crafted by a solo researcher as part of the OpenHELOS project.
If you're from a research lab or tech company interested in explainable AI, FEP or symbolic emergence — feel free to reach out.

Alexei Firssoff |
ORCID: [0009-0006-0316-116X](https://orcid.org/0009-0006-0316-116X) |
For questions, contact [a.a.firssoff@gmail.com](mailto:a.a.firssoff@gmail.com) or [🐦 @AlexFirssoff](https://x.com/AlexFirssoff).
