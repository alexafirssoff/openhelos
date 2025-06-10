# A Sheaf-Theoretic Approach to Word Sense Disambiguation

This project is an educational implementation that demonstrates how the mathematical framework of **sheaf theory** can be applied to solve the classic Natural Language Processing (NLP) problem of **Word Sense Disambiguation (WSD)**. It specifically serves as a practical, hands-on illustration for the theoretical framework detailed in **Section 4.14: 'Formal Properties and Consequences'** of the accompanying article.

The primary goal is not to compete with large-scale neural models but to illustrate an alternative, highly structured, and fully interpretable approach. It showcases how abstract mathematical concepts can provide an elegant and powerful foundation for modelling meaning and context in language.

## The Problem: Contextuality in Language

In linguistics, the meaning of a word is not a fixed, global property. Instead, it is **contextual**—it manifests differently depending on its local environment. For example, the word "bat" reveals different facets of its meaning in different contexts:

- "A **bat** flew out of the dark cave." -> Sense: `animal`
- "The baseball player broke his **bat**." -> Sense: `sports_tool`

This inherent contextuality is a central challenge in computational semantics. Attempting to define a single, global meaning for a word that is consistent across all possible contexts is often impossible. This phenomenon has deep parallels with contextuality in quantum mechanics, where the outcome of a measurement depends on the set of other measurements being performed simultaneously.

## The Solution: Sheaf Theory

**Sheaf theory** is the mathematical language of contextuality. A sheaf is a tool for systematically tracking locally-defined data and understanding how it "glues" together to form a global structure. This makes it a perfect candidate for modelling semantics.

We can formalise the problem as follows:

1.  **Base Space (X)**: We consider the set of all possible contexts as a topological space. In our simplified model, each "context word" (like `cave` or `baseball`) represents an open set in this space. The entire text or document is the whole space.

2.  **Stalk (for a point `p`)**: At any given point (representing a specific, unambiguous context), there is a set of possible meanings for our ambiguous word. This is the "stalk" over that point.

3.  **Sections (s)**: A section of the sheaf is a function `s: U -> F` that assigns a consistent meaning (or a distribution over meanings) to every point in an open set `U` (our context). In our implementation, a section is a probability distribution over senses `{sense: probability}` associated with a specific context word. For example, for the context `U = "cave"`, the section `s(U)` might be:
    `{"bat_animal": 0.98, "bat_sports_tool": 0.02}`

4.  **Gluing Axiom**: The most crucial property of a sheaf is its ability to "glue" together local sections into a global one. If we have consistent sections `s_i` defined over a collection of contexts `U_i`, they can be uniquely assembled into a single section `s` over the union of these contexts `U = ⋃ U_i`.

### Our Probabilistic Sheaf Model

This project implements a simplified, probabilistic version of this structure.

- **Knowledge Base as a Pre-Sheaf**: Our "dataset" acts as a pre-sheaf, defining sections (probability distributions) over basic open sets (the context words). Each distribution is weighted to reflect the strength of the contextual clue.

- **The `disambiguate` Method as a Gluing Mechanism**: The core of the WSD engine performs the gluing operation. When analysing a word in a sentence, it:
    a. Identifies all relevant local contexts `U_1, U_2, ..., U_n` (the context words in the window).
    b. Retrieves the corresponding local sections `s(U_1), s(U_2), ..., s(U_n)`.
    c. "Glues" them by taking a weighted sum of the probability vectors. Let `p_i(sense)` be the probability of a sense in section `i` with weight `w_i`. The global "score" for a sense is calculated as:
       `Score(sense) = Σ (w_i * p_i(sense))`
    d. Normalises these scores to produce a final global probability distribution over the senses. The sense with the highest probability is the verdict.

This process is a direct computational analogue of the sheaf-theoretic principle of moving from local data to a global conclusion.

## Key Features of the Model

- **Interpretability (Explainable AI)**: The model's primary strength. Every decision is fully transparent. The output explicitly details the contribution of each local section (context word) to the final global section (the verdict), making the reasoning process entirely auditable.

- **Modularity and Control**: The knowledge base is decoupled from the inference engine. This allows for precise, "surgical" updates to the model's understanding without requiring retraining or risking unintended side effects, a common issue with deep learning models.

- **Handling Complex Phenomena**: The model is extended to handle:
    - **Lemmatisation**: Canonically representing words to handle morphological variations.
    - **Context Weighting**: Differentiating between strong and weak contextual indicators.
    - **Negation**: A simple mechanism to invert or ignore the influence of a local section when it is negated.

## Example of Gluing in Action

Consider the sentence: "A baseball player broke his **bat** in the **cave**."

1.  **Local Contexts Found**: `baseball` and `cave`.
2.  **Local Sections Retrieved**:
    - `s("baseball")` -> `{"animal": 0.01, "tool": 0.99}` with weight `w=2.0`.
    - `s("cave")` -> `{"animal": 0.98, "tool": 0.02}` with weight `w=1.5`.
3.  **Gluing (Weighted Summation)**:
    - `Score(animal)` = (2.0 * 0.01) + (1.5 * 0.98) = 0.02 + 1.47 = **1.49**
    - `Score(tool)` = (2.0 * 0.99) + (1.5 * 0.02) = 1.98 + 0.03 = **2.01**
4.  **Global Section (Normalisation)**:
    - The total score is 1.49 + 2.01 = 3.50.
    - `P(animal)` = 1.49 / 3.50 ≈ 0.43
    - `P(tool)` = 2.01 / 3.50 ≈ 0.57
5.  **Verdict**: `sports_tool`, but with relatively low confidence, reflecting the conflicting nature of the evidence.

This project, while educational, provides a powerful illustration of how principled mathematical structures can lead to robust, efficient, and—most importantly—interpretable AI systems for complex linguistic tasks.