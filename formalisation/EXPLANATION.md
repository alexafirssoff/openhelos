# Formalisation of the Core Data Structure S

This directory contains the formal specification and key proofs for the core data structure of the framework, implemented in the **Coq proof assistant**.

The purpose of this formalisation is to mathematically verify the foundational properties of our proposed cognitive substrate, ensuring its integrity and lossless nature before its implementation in software.

---

### File: `FrameworkFormalisation.v`

This file defines the central hierarchical data structure, `S_list`, and proves a critical isomorphism.

#### 1. Definition of the Hierarchical Structure (`S_list`)

The structure `S_list` is defined inductively to represent hierarchical, compositional thought structures. It is built upon a generic parameter `I`, representing any atomic unit of information (a symbol, a sensory input, etc.).

An `S_list` can be constructed in one of three ways:

- **`C1_list`**: A base case, forming a primitive structure from two atomic elements. This can be thought of as the simplest "link" or "association".
- **`C2_list`**: A compositional constructor. It takes a list of existing atoms (`I`) and sub-structures (`S_list`) to form a new, more complex structure. This rule allows for the recursive and hierarchical assembly of knowledge.
- **`C3_list`**: An alternative recursive constructor for extending an existing structure.

This inductive definition provides a formal grammar for building arbitrarily complex, tree-like representations of knowledge from a basic set of components.

#### 2. The Isomorphism Proof (`Slist_iso_RHS_v2`)

The core of this file is the proof that our `S_list` type is **isomorphic** to its "deconstructed" representation (`RHS_list_type`), which is a mathematical sum of products.

**What does this isomorphism prove?**

In practical terms, it formally guarantees that:

1.  **Lossless Deconstruction:** Any complex structure (`S_list`) can be broken down into its fundamental constituent parts without any loss of information.
2.  **Perfect Reconstruction:** Those parts can be reassembled back into the exact original structure.

This property is not merely a technical detail; it is the **mathematical cornerstone of the framework's native interpretability**. Because we can provably deconstruct any "thought" into its components, we can inspect, analyse, and understand its structure. This stands in stark contrast to opaque "black-box" models where such analysis is impossible.

The proof is established by defining `to` and `from` functions and then proving that their compositions are identity functions (`to ∘ from = id` and `from ∘ to = id`).

The structure of the isomorphism proof can be visualised as follows:

```
iso S_list RHS_list_type
├── to : S_list → RHS_list_type
│ └── slist_to_rhs
├── from : RHS_list_type → S_list
│ └── rhs_to_slist
├── to_from : ∀ (b : RHS_list_type), to (from b) = b
│ └── slist_to_rhs_circ_rhs_to_slist
└── from_to : ∀ (a : S_list), from (to a) = a
└── rhs_to_slist_circ_slist_to_rhs
```

### How to Verify

To check the proofs, you will need the Coq proof assistant (version 8.x) installed. You can then compile and verify the file from your terminal:

```sh
coqc SStructureIsomorphism.v
```

A successful compilation with no errors serves as machine-checked verification of the claims made in the file.

