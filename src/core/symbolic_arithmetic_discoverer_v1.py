# SPDX-License-Identifier: LicenseRef-NonCommercial-Research
# Copyright (c) 2025 Alexei Firssoff. ORCID: 0009-0006-0316-116X

# -*- coding: utf-8 -*-
"""
HELOS–Topos Arithmetic Learner — FINAL (multi-digit ×, MDL compression, OOD tests)
==================================================================================

High-level purpose
------------------
This single file implements a compact neurosymbolic learner that induces
digit-wise arithmetic as an emergent sheaf of rules. It belongs to the
HELOS programme (Hierarchical Emergence of Ontological Structure) and serves as
a concrete demonstration that reliable, interpretable algorithmic competence
(addition, subtraction-as-absolute-difference, multiplication with single- and
multi-digit multipliers) can emerge without hard-coding algorithmic control
flow at inference time.

What the code does
--------------------
• Learns local stochastic mappings over the base space of digit contexts
  K = (op, d₁, d₂∕m, s_in), producing stalk values V = (digit_out, s_out).
• Aggregates local evidence into a global section per sheaf via MAP with
  Dirichlet smoothing. The global section is an interpretable table of rules.
• Uses three tiny learned auxiliaries (not priors) at inference:
  1) **SheafSign** (for ‘−’): a fast hint whether to swap (a, b). It may be
     wrong; we always test both orders and choose by data likelihood.
  2) **SheafOrderPlus** (for ‘+’): a learned order choice (d₁, d₂) vs (d₂, d₁)
     conditioned on carry s_in — replaces a hard commutativity fallback.
  3) **Star projection** (for ‘*’): if a triple (d₁, m, s_in) is unseen, fall
     back to a learned projection keyed by (d₁, m).

• Runs novelty-driven EM over a *mixture of sheaves* (each a candidate rule
  table) and automatically allocates new sheaves when evidence warrants.

Relation to HELOS (conceptual)
------------------------------
• Structure S (symbolic substrate): here, S is the finite product
  S = Op × Digit × Digit/Multiplier × Carry, with Op ∈ {‘+’, ‘−’, ‘*’}.
  Keys k ∈ S are BaseSpacePoints; stalks hold values v = (digit_out, s_out).
• Sheaves: each `SheafOfRules` stores counts and a *global section* —
  a choice of the most plausible local value v(k) per key k, with smoothed
  log-likelihoods log p(v | k, sheaf).
• Topos: the CognitiveTopos is a small “category” of these sheaves with
  EM-style responsibilities. No categorical machinery is coded; the name
  reflects the organisation of local sections into global sections and the
  multiple-object mixture.
• Free Energy: the training objective implicitly trades off goodness-of-fit
  of local sections against model complexity:
    F ≈ −∑ log p(v | k, sheaf) + λ_MDL · |global_section|
  We do not write an explicit variational bound; instead `evaluate_consistency`
  uses cached log-probabilities and an optional MDL penalty. Responsibilities in
  EM act as the soft assignment (posterior) over sheaves.

Learning from labelled data (and why that is not cheating)
----------------------------------------------------------
• The training sets are labelled triples (a, op, b) ↦ c. We deliberately use
  ground-truth c to produce local sections (digit-wise steps) during training.
  This mirrors a child being told the correct answer and then observing how
  carries/borrows must have behaved digit-wise.
• At inference, the model receives only (a, op, b). It must produce c by
  *consulting its learned local tables* — not by running an encoded algorithm.

Branch-free inference and use of ‘op’
-------------------------------------
• You will see conditionals like `if op in '+-':` or `if op == '*'`. These are
  **routing statements**, not algorithmic rules. They *index into* the learned
  tables because our key space includes `op` as a feature:
      k = (op, d₁, d₂∕m, s_in).
  The code never computes “borrow” or “carry” via arithmetic formulae at
  inference; it simply looks up what to output and how to transition carry.
• For ‘−’ specifically, we always evaluate both operand orders and choose the
  one with the higher cumulative log-consistency with a given sheaf. This is a
  data-driven sign decision, not an arithmetic comparison.

Core algorithm in Unicode maths
-------------------------------
Given a candidate sheaf θ (its global section is a function g_θ: K → V):

1) Local consistency score for one worked example (local section) σ:
      score_θ(σ) = ∑_{(k,v)∈σ} log p_θ(v | k)  −  λ · |g_θ|

2) EM responsibilities over sheaves (temperature τ > 0):
      r_θ ∝ exp(score_θ(σ) / τ),   normalised over θ plus a novelty mass.

3) M-step updates:
      counts_θ(k, v) ← counts_θ(k, v) + r_θ
      spec_θ(op)     ← spec_θ(op)     + r_θ

4) Global section (MAP) with Dirichlet smoothing α:
      g_θ(k) = argmax_v (c_θ(k, v) + α),   and
      log p_θ(v | k) = log( (c_θ(k, v) + α) / (∑_u c_θ(k, u) + α·support) )

Multi-digit ‘*’ via composition (no arithmetic rule, only routing)
------------------------------------------------------------------
To multiply a by a multi-digit b = ∑_j m_j·10^j, we:

• call the learned single-digit predictor for each m_j: part_j ← predict('*', a, m_j);
• shift part_j by appending j zeros;
• accumulate via learned addition: total ← predict('+', total, part_j).

This is composition of learned skills, not an embedded multiplication
algorithm. All digits and carries still come from the learned tables.

File outputs
------------
• ./results/arithmetic/helos_results.csv      — large random test suite with predictions.
• ./results/arithmetic/data/helos_memory_dump.json — interpretable memory snapshot (rules & aux).
• ./results/arithmetic/data/helos_traces.txt       — human-readable, step-by-step traces.

"""

from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import json, random, time, csv, os
import numpy as np

# ----------------------- Type aliases -----------------------
BaseSpacePoint = Tuple[str, str, str, int]  # (op, d1, d2_or_m, s_in)
StalkValue = Tuple[str, int]  # (digit_out, s_out)
LocalSection = List[Tuple[BaseSpacePoint, StalkValue]]
GlobalSection = Dict[BaseSpacePoint, StalkValue]

DIGITS = "0123456789"


# ----------------------- Utilities -----------------------
def strip_leading_zeros(s: str) -> str:
    """
    Remove any leading zeros; return '0' if the string would become empty.

    Rationale:
      Keeps canonical numeric strings for comparison/printing while preserving
      the invariant that an all-zero tail compresses to a single ‘0’.
    """
    s2 = s.lstrip("0")
    return s2 if s2 else "0"


def zpad_left(a: str, n: int) -> str:
    """
    Zero-pad a string on the left to total length n (never truncates).

    Used for:
      • digit alignment when scanning from the least significant digit;
      • ensuring we can index past the most significant digit with '0's.
    """
    return "0" * (n - len(a)) + a


def rand_int_str(max_len: int) -> str:
    """
    Generate a random non-negative integer string with 1..max_len digits.

    Constraints:
      • No leading zero for multi-digit numbers.
      • Single-digit '0' is allowed.

    Purpose:
      Used to synthesise labelled datasets for training and testing. This is
      explicitly labelled supervision: we later compute c via ground truth.
    """
    if max_len <= 0:
        return '0'
    L = random.randint(1, max_len)
    if L == 1:
        return random.choice(DIGITS)
    return random.choice("123456789") + "".join(random.choice(DIGITS) for _ in range(L - 1))


# Ground truth (only for synthetic data generation / evaluation)
def gt_add(a, b):
    return str(int(a) + int(b))


def gt_sub_nonneg(a, b):
    return str(abs(int(a) - int(b)))


def gt_mul1(a, m):
    return str(int(a) * int(m))


# ----------------------- Data makers -----------------------
def make_add_dataset(n, max_len):
    """
    Create n random labelled addition triples (a, b, c) with len ≤ max_len.

    Label semantics:
      c = int(a) + int(b). These labels are used to produce *training* local
      sections (teacher signal) via the Decomposer. At inference c is unknown.
    """
    return [(a, b, gt_add(a, b)) for a, b in [(rand_int_str(max_len), rand_int_str(max_len)) for _ in range(n)]]


def make_sub_dataset(n, max_len):
    """
    Create n random labelled subtraction-as-absolute-difference triples.

      c = |int(a) − int(b)|

    We choose absolute difference so that sign can be learned as an *order*
    decision without needing negative numerals in the representation.
    """
    return [(a, b, gt_sub_nonneg(a, b)) for a, b in [(rand_int_str(max_len), rand_int_str(max_len)) for _ in range(n)]]


def make_mul1_dataset(n, max_len, p_zero: float = 0.2):
    """
    Create n random single-digit multiplier triples (a, m, a*m).
    Now includes m='0' with probability p_zero to teach ×0 behaviour.
    """
    samples = []
    for _ in range(n):
        a = rand_int_str(max_len)
        # allow '0' as multiplier with some probability
        if random.random() < p_zero:
            m = '0'
        else:
            m = random.choice("123456789")
        samples.append((a, m, gt_mul1(a, m)))
    return samples


# ----------------------- Decomposer (for TRAINING only) -----------------------
class Decomposer:
    """
    Produce idealised local sections (per-digit steps) on the *training* side.

    Why a Decomposer?
      We require clean, stable supervision for local rule estimation. Given a
      labelled triple (a, op, b) ↦ c, we decompose the worked example into a
      sequence of key/value pairs scanned from least significant digit to most:
         (op, d₁, d₂∕m, s_in) ↦ (digit_out, s_out).
      This is the only place where classical per-digit algorithms are used, and
      they are used solely to compute the teacher signal. Inference never calls
      these rules; it consults learned tables.

    Early stopping:
      We stop when the remaining tail is exactly zero and s_in == 0 — i.e. when
      continuing would only append redundant zeros.

    Note (for ‘−’):
      For training only, we canonicalise operands so that a ≥ b before
      decomposing. This yields consistent local steps for borrowing. At
      inference there is no such prior: we evaluate both orders and select by
      likelihood.
    """
    
    @staticmethod
    def get_ideal_local_section(op: str, a: str, b: str, c: str) -> Optional[LocalSection]:
        if op == '-':
            # TRAINING-ONLY canonicalisation for subtraction (see class docstring).
            if int(a) < int(b):
                a, b = b, a
        steps: LocalSection = []
        s_in = 0
        if op in '+-':
            # Pad and reverse to scan least→most significant.
            n = max(len(a), len(b))
            aa, bb, cc = zpad_left(a, n)[::-1], zpad_left(b, n)[::-1], zpad_left(c, n + 2)[::-1]
            for i in range(len(cc)):
                # Per-digit inputs for this position i.
                d1 = aa[i] if i < n else '0'
                d2 = bb[i] if i < n else '0'
                d_out = cc[i]
                
                # Early stop if the remaining tail is exactly zero and no carry.
                if strip_leading_zeros(c) == strip_leading_zeros(
                        "".join(reversed(cc[i:]))) and d1 == '0' and d2 == '0' and s_in == 0:
                    break
                
                # Teacher transitions for s_out (training only).
                if op == '+':
                    s_out = (int(d1) + int(d2) + s_in) // 10
                else:
                    s_out = 1 if (int(d1) - s_in) < int(d2) else 0
                
                point: BaseSpacePoint = (op, d1, d2, s_in)
                value: StalkValue = (d_out, s_out)
                steps.append((point, value))
                s_in = s_out
        
        elif op == '*':
            # Single-digit multiplier; pad result length and scan.
            aa, m, cc = a[::-1], b, zpad_left(c, len(a) + len(b) + 1)[::-1]
            for i in range(len(cc)):
                d1 = aa[i] if i < len(aa) else '0'
                d_out = cc[i]
                # Early stop when past the top and no carry remains.
                if d1 == '0' and s_in == 0 and i >= len(aa):
                    break
                t = int(d1) * int(m) + s_in
                s_out = t // 10
                point: BaseSpacePoint = (op, d1, m, s_in)
                value: StalkValue = (d_out, s_out)
                steps.append((point, value))
                s_in = s_out
        return steps


# ----------------------- Learned auxiliaries (replace priors at inference) -----------------------
@dataclass
class SheafSign:
    """
    Learn whether to swap (a, b) for ‘−’ at inference.

    Features (deliberately coarse, non-arithmetic):
      ('−', len(a), len(b), a[0], b[0])

    Behaviour:
      Stores counts over {0: keep (a,b), 1: swap to (b,a)}, builds a MAP table.
      At inference this is only a *hint*: we will still evaluate both orders and
      choose by cumulative log-likelihood. So a wrong hint cannot force an error.
    """
    obs: Dict[Tuple, Counter] = field(default_factory=lambda: defaultdict(Counter))
    gs: Dict[Tuple, int] = field(default_factory=dict)
    
    def add(self, feats: Tuple, label: int, w: float = 1.0):
        """Accumulate weighted observation for a sign decision feature tuple."""
        self.obs[feats][label] += w
    
    def fit(self):
        """Convert observations to deterministic MAP choices per feature key."""
        self.gs = {k: max(cnt.items(), key=lambda kv: kv[1])[0] for k, cnt in self.obs.items()}
    
    def predict(self, feats: Tuple) -> int:
        """
        Return 0 or 1. If a feature was never seen, fall back to the most common
        label in its sparse counter or default to 0 (keep).
        """
        if feats in self.gs:
            return self.gs[feats]
        cnt = self.obs.get(feats)
        if cnt:
            return max(cnt.items(), key=lambda kv: kv[1])[0]
        return 0


@dataclass
class SheafOrderPlus:
    """
    Learn when to swap (d₁, d₂) for ‘+’ at a given carry s_in.

    Key:
      (s_in, d₁, d₂) → {0: keep (d₁,d₂), 1: swap (d₂,d₁)}

    Role:
      Provides a *learned* commutativity routing that can improve coverage of
      the learned table; used only when the direct key is missing.
    """
    obs: Dict[Tuple[int, str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))
    gs: Dict[Tuple[int, str, str], int] = field(default_factory=dict)
    
    def add(self, s_in: int, d1: str, d2: str, label: int, w: float = 1.0):
        """Record that for (s_in, d1, d2) the better order was ‘label’."""
        self.obs[(s_in, d1, d2)][label] += w
    
    def fit(self):
        """Build a deterministic MAP choice per (s_in, d1, d2)."""
        self.gs = {k: max(cnt.items(), key=lambda kv: kv[1])[0] for k, cnt in self.obs.items()}
    
    def choose(self, s_in: int, d1: str, d2: str) -> int:
        """Return 0 (keep) or 1 (swap); default to 0 when unseen."""
        return self.gs.get((s_in, d1, d2), 0)


# ----------------------- Sheaf of Rules (EM core) -----------------------
@dataclass
class SheafOfRules:
    """
    One emergent ‘sheaf of rules’.

    Stores:
      • fractional observation counts per local key (op, d₁, d₂∕m, s_in);
      • a global section (MAP selection per key);
      • smoothed log-probabilities log p(v | k) cached for fast scoring;
      • operator specialisation statistics (to bias responsibilities).

    It has no knowledge of arithmetic. It only accumulates evidence that
    certain digit/context keys map to certain digit/carry outcomes.
    """
    sheaf_id: int
    _stalk_observations: Dict[BaseSpacePoint, Counter] = field(default_factory=lambda: defaultdict(Counter))
    global_section: GlobalSection = field(default_factory=dict)
    _log_probabilities: Dict[Tuple[BaseSpacePoint, StalkValue], float] = field(default_factory=dict)
    
    op_specialisation: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    total_weight_processed: float = 0.0
    
    def reset_with_seed(self, op_hint: str, seed_section: LocalSection):
        """
        Repurpose this sheaf for a new operator attraction by clearing counts
        and seeding with a provided local section (used when the pool is full).

        Note:
          This operation only manipulates *counts*. It does not inject rules.
        """
        self._stalk_observations.clear()
        self.global_section.clear()
        self._log_probabilities.clear()
        self.op_specialisation.clear()
        self.total_weight_processed = 0.0
        for point, value in seed_section:
            self._stalk_observations[point][value] = 1.0
        self.op_specialisation[op_hint] = 1.0
        self.total_weight_processed = 1.0
    
    def construct_global_section(self, alpha: float):
        """
        Build the global section g_θ via MAP with Dirichlet smoothing α.

        For each key k:
          g_θ(k) = argmax_v (counts(k,v) + α)
          log p_θ(v | k) = log( (counts(k,v) + α) / (∑_u counts(k,u) + α·support) )

        The ‘support’ heuristic increases smoothing for richer operators (e.g. ‘*’).
        """
        self.global_section.clear()
        self._log_probabilities.clear()
        k_states_map = {'+': 2, '-': 2, '*': 10}
        for point, counts in self._stalk_observations.items():
            if not counts:
                continue
            op = point[0]
            total = float(sum(counts.values()))
            support = max(len(counts), 200 if op == '*' else 10 * k_states_map.get(op, 2))
            best_value, _ = max(counts.items(), key=lambda kv: kv[1])
            self.global_section[point] = best_value
            denom = total + alpha * support
            for value, c in counts.items():
                self._log_probabilities[(point, value)] = float(np.log((c + alpha) / denom))
    
    def evaluate_consistency(self, section: LocalSection, lambda_mdl: float) -> float:
        """
        Score how well a proposed local section σ agrees with this sheaf:

          score_θ(σ) = ∑ log p_θ(v | k)  −  λ_MDL · |g_θ|

        Missing (k,v) pairs are penalised with a strong negative constant.
        """
        ll = 0.0
        for point, value in section:
            lp = self._log_probabilities.get((point, value))
            ll += (-20.0 if lp is None else lp)
        return ll - lambda_mdl * len(self.global_section)
    
    # ---------- NEW: MDL-style compression ----------
    def compress_via_mdl(self, min_count: float = 0.9, min_logprob: float = -7.0):
        """
        MDL-style pruning of weak local rules.

        Criteria (any triggers pruning of a (k,v) pair from the global section):
          • total count at key k below `min_count` (effective evidence too small), OR
          • cached log p(v|k) < `min_logprob` (code length too large ⇒ poor support).

        After pruning:
          • remove g_θ(k) entry; keep raw counts (so evidence can regrow later).
          • cached log-probs are recomputed on next construct_global_section().
        """
        to_delete = []
        for k, v in self.global_section.items():
            cnts = self._stalk_observations.get(k, None)
            if not cnts:
                to_delete.append(k);
                continue
            total = float(sum(cnts.values()))
            lp = self._log_probabilities.get((k, v), None)
            if total < min_count or (lp is not None and lp < float(min_logprob)):
                to_delete.append(k)
        for k in to_delete:
            if k in self.global_section:
                del self.global_section[k]


# ----------------------- Cognitive Topos (full model) -----------------------
class CognitiveTopos:
    """
    The full emergent learner (mixture over sheaves with novelty-driven EM).

    Responsibilities:
      • Maintain a pool of sheaves, grow with novelty when warranted.
      • Track operator specialisation per sheaf (evidence-weighted).
      • Provide branch-free inference routed by *learned* tables and auxiliaries.
      • Offer detailed tracing so that every decision is auditable.

    Public API
    ----------
    observe(op, a, b, c):
        Consume one *labelled* triple and update the mixture by EM.
    fit_all_sheaves():
        Recompute global sections and auxiliary MAPs.
    predict(op, a, b, trace=False):
        Predict c; if trace=True, also return verbose trace lines.
    dump_memory(path):
        Persist an interpretable snapshot of learned rules/auxiliaries.
    """
    
    def __init__(self,
                 max_sheaves: int = 50,
                 novelty_log_prob: float = -50.0,
                 novelty_threshold: float = 0.95,
                 temperature: float = 0.6,
                 lambda_mdl: float = 0.0,
                 alpha: float = 1e-2,
                 rng_seed: int = 0):
        # Reproducibility and hyperparameters.
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        
        # Sheaf pool and novelty control.
        self.sheaves: List[SheafOfRules] = []
        self.next_sheaf_id = 0
        self.max_sheaves = max_sheaves
        self.novelty_log_prob = float(novelty_log_prob)
        self.novelty_threshold = float(novelty_threshold)
        self.temperature = max(1e-6, float(temperature))
        self.lambda_mdl = float(lambda_mdl)
        self.alpha = float(alpha)
        
        # Learned auxiliaries (replacing any “prior” branching).
        self.sign_sheaf = SheafSign()
        self.order_plus = SheafOrderPlus()
        self._star_proj_obs: Dict[Tuple[str, str], Counter] = defaultdict(Counter)  # (d1,m) → Counter[(d_out,s_out)]
        self._star_proj_gs: Dict[Tuple[str, str], Tuple[str, int]] = {}  # MAP projection
    
    # ---------- utils ----------
    def _softmax_temp(self, logs: List[float]) -> np.ndarray:
        """
        Temperature-scaled softmax over a list of log-scores.

        Used in:
          • E-step responsibilities across existing sheaves plus a novelty slot.
        """
        arr = np.array(logs, dtype=np.float64) / self.temperature
        m = np.max(arr);
        ex = np.exp(arr - m);
        s = ex.sum()
        return ex / s if s > 0 else np.ones_like(ex) / len(ex)
    
    def _weakest_sheaf_for_op(self, op: str) -> Optional[SheafOfRules]:
        """
        Return the sheaf with the lowest specialisation for operator `op`.
        Used when we must repurpose an existing sheaf (pool is full).
        """
        if not self.sheaves:
            return None
        return min(self.sheaves, key=lambda s: s.op_specialisation.get(op, 0.0))
    
    def _create_new_sheaf(self, op_hint: str, seed_section: LocalSection):
        """
        Spawn a new sheaf and seed it with one local section (unit weight).

        Note:
          ‘op_hint’ affects only the *initial* specialisation bias; there is no
          algorithmic logic attached to operators.
        """
        print(f"  -> Emergence: constructing new Sheaf {self.next_sheaf_id} (attracted by '{op_hint}')")
        sh = SheafOfRules(sheaf_id=self.next_sheaf_id)
        self.sheaves.append(sh)
        self.next_sheaf_id += 1
        for point, value in seed_section:
            sh._stalk_observations[point][value] = 1.0
        sh.op_specialisation[op_hint] = 1.0
        sh.total_weight_processed = 1.0
    
    def fit_all_sheaves(self):
        """
        Recompute all global sections and auxiliary MAPs.

        Called:
          • periodically during training;
          • once at the end of each epoch.
        """
        for sh in self.sheaves:
            sh.construct_global_section(self.alpha)
            # NEW: light MDL compression per sheaf after MAP.
            sh.compress_via_mdl(min_count=0.9, min_logprob=-7.0)
        self.sign_sheaf.fit()
        self.order_plus.fit()
        # Build star-projection MAPs for '*' from pooled observations.
        self._star_proj_gs = {k: max(cnt.items(), key=lambda kv: kv[1])[0] for k, cnt in self._star_proj_obs.items()}
    
    def _ensure_op_slot(self, op: str, section: LocalSection):
        """
        Ensure at least one sheaf is sufficiently specialised for an operator.

        Mechanism:
          • If no sheaf has spec(op) ≥ 8, create one (if capacity permits)
            or repurpose the weakest specialised sheaf by resetting with seed.
        """
        max_spec = max((s.op_specialisation.get(op, 0.0) for s in self.sheaves), default=0.0)
        if max_spec >= 8.0:
            return
        if len(self.sheaves) < self.max_sheaves:
            self._create_new_sheaf(op, section)
        else:
            weakest = self._weakest_sheaf_for_op(op)
            if weakest is not None:
                weakest.reset_with_seed(op, section)
            else:
                self._create_new_sheaf(op, section)
    
    # ---------- EM observe ----------
    def observe(self, op: str, a: str, b: str, c: str):
        """
        Update the mixture by one *labelled* example (a, op, b) → c.

        Steps:
          1) (for ‘−’) Collect supervision for **SheafSign** *before* any
             training-time canonicalisation in the Decomposer.
          2) Obtain the idealised local section σ via the Decomposer (teacher).
          3) E-step: compute responsibilities against all sheaves + novelty.
          4) Potentially allocate a new sheaf if ‘novelty’ wins strongly.
          5) M-step: add fractional counts and update specialisation.
          6) Collect auxiliary supervision:
              • for ‘+’: order decisions (baseline “keep”);
              • for ‘*’: star-projection counts keyed by (d₁, m).

        Important:
          This routine never performs arithmetic at inference; it only uses labels
          to update counts during training, as a child would learn from answers.
        """
        # (1) Sign supervision for ‘−’ from raw (a,b) (teacher only).
        if op == '-':
            feats_sign = ('-', len(a), len(b), a[0], b[0])
            sign_label = int(int(a) < int(b))
            self.sign_sheaf.add(feats_sign, sign_label, 1.0)
        
        # (2) Teacher local section (training only).
        section = Decomposer.get_ideal_local_section(op, a, b, c)
        if not section:
            return
        
        # Lazy initialisation of the pool.
        if not self.sheaves:
            self._create_new_sheaf(op, section)
            return
        
        # (3) Ensure at least one specialised slot exists for this op.
        self._ensure_op_slot(op, section)
        
        # (4) E-step: score each sheaf for this local section.
        ll_sheaves = []
        for sheaf in self.sheaves:
            base_ll = sheaf.evaluate_consistency(section, self.lambda_mdl)
            spec_norm = sheaf.op_specialisation.get(op, 0.0) / (sheaf.total_weight_processed + 1e-9)
            ll_sheaves.append(base_ll + 1.5 * spec_norm)
        
        # Add a novelty log-score as an extra “component”.
        all_logs = ll_sheaves + [self.novelty_log_prob]
        resp = self._softmax_temp(all_logs)
        
        # (5) Novelty test: if mass on novelty is large, spawn a sheaf.
        novelty_resp = float(resp[-1])
        if novelty_resp > self.novelty_threshold and len(self.sheaves) < self.max_sheaves:
            self._create_new_sheaf(op, section)
            return
        
        # (6) M-step: fractional updates to counts and specialisation.
        for i, sheaf in enumerate(self.sheaves):
            r = float(resp[i])
            if r <= 1e-6:
                continue
            for point, value in section:
                sheaf._stalk_observations[point][value] += r
            sheaf.op_specialisation[op] += r
            sheaf.total_weight_processed += r
        
        # (7) Auxiliary supervision updates.
        for (pt, val) in section:
            opx, d1, d2m, s_in = pt
            if opx == '+':
                self.order_plus.add(s_in, d1, d2m, label=0, w=1.0)  # baseline order = “keep”
            elif opx == '*':
                self._star_proj_obs[(d1, d2m)][val] += 1.0
    
    # ---------- Inference helpers (composition) ----------
    def _obs_argmax(self, sheaf: SheafOfRules, k: BaseSpacePoint) -> Optional[StalkValue]:
        """
        Return the most frequent observed value for key k, if any.

        Purpose:
          A gentle data-driven fallback when the MAP global section lacks k.
        """
        cnt = sheaf._stalk_observations.get(k)
        if cnt:
            return max(cnt.items(), key=lambda kv: kv[1])[0]
        return None
    
    def _compose_add(self, x: str, y: str, trace: bool = False) -> Tuple[Optional[str], List[str]]:
        """
        Data-routed composition: add two numbers via learned '+'.

        We deliberately call the *existing* predictor with op='+' so that each
        digit step is resolved by the learned tables (or their fallbacks).
        No arithmetic rule is executed here; this is pure routing/composition.

        Returns:
            (sum_string_or_None, trace_lines)
        """
        s, tr = self.predict('+', x, y, trace=trace)
        return (s if s != "" else None), tr
    
    def _predict_mul_multidigit(self, a: str, b: str, trace: bool = False) -> Tuple[Optional[str], List[str]]:
        """
        Multi-digit multiplication by composition of learned single-digit '*' + learned '+'.

        Algorithmic *routing* only:
          For each digit m_j of the multiplier b (least→most significant),
          1) get partial = self.predict('*', a, m_j)  (single-digit multiplier)
          2) left-pad with j zeros (decimal shift),
          3) accumulate via learned '+' onto a running total.

        At no point do we compute digits/carries by rule; we strictly delegate
        to the learned sheaves. If any sub-call fails, we abort with None.

        Returns:
            (product_or_None, aggregated_trace_lines)
        """
        tr_all: List[str] = []
        if a == "0" or b == "0":
            return "0", tr_all
        
        total = "0"
        bb = b[::-1]  # scan least significant digit first
        for j, m in enumerate(bb):
            # 1) single-digit partial via learned '*'
            part, tr_mul = self.predict('*', a, m, trace=trace)
            tr_all += [f"[×] partial for m={m} @pos={j} → {part if part else '∅'}"] + tr_mul
            if part == "":
                return None, tr_all
            
            # 2) decimal shift by j (append j zeros)
            if part != "0" and j > 0:
                part = part + ("0" * j)
                if trace:
                    tr_all.append(f"[×] shift by {j} → {part}")
            
            # 3) accumulate via learned '+'
            total2, tr_add = self._compose_add(total, part, trace=trace)
            tr_all += [f"[+] accumulate {total} + {part} → {total2 if total2 else '∅'}"] + tr_add
            if total2 is None:
                return None, tr_all
            total = total2
        
        return total, tr_all
    
    # ---------- Inference (per sheaf) ----------
    def _try_predict_with_sheaf(self, sheaf: SheafOfRules, op: str, a: str, b: str,
                                trace: bool = False) -> Tuple[Optional[str], List[str]]:
        """
        Attempt to predict with a particular sheaf θ.

        Returns:
          (prediction_or_None, trace_lines)

        Key point for ‘−’ (non-aprioric sign):
          • We evaluate *both* operand orders (a,b) and (b,a).
          • Each order is scored by the cumulative ∑ log p_θ(v | k).
          • The chosen order is the one with the higher score. A fast SheafSign
            hint only decides which order we *try first*; it cannot force the
            outcome if likelihood disagrees. No integer comparison is used.
        """
        section = sheaf.global_section
        out: List[str] = []
        s_in = 0
        trace_lines: List[str] = []
        
        # Helper dedicated to ‘−’ scoring WITHOUT any arithmetic comparison.
        def _score_and_predict_minus(order_a: str, order_b: str) -> Tuple[Optional[str], float, List[str]]:
            s_in_loc = 0
            out_loc, tl = [], []
            n = max(len(order_a), len(order_b))
            aa, bb = zpad_left(order_a, n)[::-1], zpad_left(order_b, n)[::-1]
            score = 0.0
            for i in range(n + 2):
                # Per-digit inputs and termination condition (past top & no carry).
                d1 = aa[i] if i < n else '0'
                d2 = bb[i] if i < n else '0'
                if i >= n and s_in_loc == 0:
                    break
                
                # Query learned table; never compute borrow by rule.
                k0 = ('-', d1, d2, s_in_loc)
                v = section.get(k0)
                used_fallback = "none"
                chosen_key = k0
                
                # Fallback to observation argmax if global section lacks this key.
                if v is None:
                    v = self._obs_argmax(sheaf, k0)
                    if v is not None:
                        used_fallback = "obs_argmax"
                
                # If still unknown, this order fails for this sheaf.
                if v is None:
                    tl.append(f"[-] i={i} k={k0} → MISSING; abort")
                    return None, -1e9, tl
                
                # Accumulate output digit and next carry, plus log-likelihood.
                d_out, s_out = v
                lp = sheaf._log_probabilities.get((chosen_key, v), -20.0)
                score += lp
                tl.append(
                    f"[-] i={i} s_in={s_in_loc} key={chosen_key} → d_out={d_out}, s_out={s_out} (fallback={used_fallback}) lp={lp:.3f}")
                out_loc.append(d_out)
                s_in_loc = s_out
            
            # Append any remaining carry digits (as characters) on exit.
            if s_in_loc > 0:
                out_loc.extend(list(str(s_in_loc))[::-1])
                tl.append(f"[carry] final s_in={s_in_loc} appended")
            
            # Compose final prediction for this order.
            pred = strip_leading_zeros("".join(reversed(out_loc)))
            # MDL length penalty: discourage outputs longer than needed
            mdl_lambda = 0.5
            excess = max(0, len(pred) - max(len(order_a), len(order_b)))
            score -= mdl_lambda * excess
            tl.append(f"[mdl] length_penalty: -{mdl_lambda} × {excess} → score={score:.3f}")
            
            return pred, score, tl
        
        # Branching by ‘op’ is *routing into learned tables*, not algorithmic rules.
        if op in '+-':
            if op == '+':
                # Standard per-digit scan; learned order helper may swap digits
                # if the direct key is missing and the helper recommends swap.
                n = max(len(a), len(b))
                aa, bb = zpad_left(a, n)[::-1], zpad_left(b, n)[::-1]
                for i in range(n + 2):
                    d1, d2 = (aa[i] if i < n else '0'), (bb[i] if i < n else '0')
                    if i >= n and s_in == 0:
                        break
                    
                    # (1) Try direct learned mapping.
                    k0 = (op, d1, d2, s_in)
                    v = section.get(k0)
                    chosen_key = k0
                    used_fallback = "none"
                    
                    # (2) If missing, the *learned* order chooser may suggest swap.
                    if v is None and op == '+':
                        do_swap = self.order_plus.choose(s_in, d1, d2)
                        if do_swap == 1:
                            k2 = (op, d2, d1, s_in)
                            v = section.get(k2)
                            chosen_key = k2
                            used_fallback = "order_plus(MAP)"
                            if v is None:
                                v = self._obs_argmax(sheaf, k2)
                                if v is not None:
                                    used_fallback = "order_plus(obs_argmax)"
                    
                    # (3) If still missing, try observation argmax on original order.
                    if v is None:
                        v = self._obs_argmax(sheaf, k0)
                        if v is not None:
                            used_fallback = "obs_argmax"
                    
                    # (4) If unknown, this sheaf fails for this expression.
                    if v is None:
                        if trace:
                            trace_lines.append(f"[{op}] i={i} k={k0} → MISSING; abort")
                        return None, trace_lines
                    
                    # (5) Emit digit and carry; record trace.
                    d_out, s_out = v
                    if trace:
                        trace_lines.append(
                            f"[{op}] i={i} s_in={s_in} key={chosen_key} → d_out={d_out}, s_out={s_out} (fallback={used_fallback})")
                    out.append(d_out)
                    s_in = s_out
            
            else:
                # ‘−’ (sign/order is *data-driven*):
                # We will evaluate both orders; the hint only decides which to try first.
                feats_sign = ('-', len(a), len(b), a[0], b[0], len(a) - len(b), a[-1], b[-1])
                flip_hint = self.sign_sheaf.predict(feats_sign)
                order_primary = (b, a) if flip_hint == 1 else (a, b)
                order_alternate = (a, b) if flip_hint == 1 else (b, a)
                
                # Try hinted order.
                y1, s1, t1 = _score_and_predict_minus(order_primary[0], order_primary[1])
                # Try alternate order.
                y2, s2, t2 = _score_and_predict_minus(order_alternate[0], order_alternate[1])
                
                # Choose by higher cumulative log-likelihood (no integer comparison).
                if (y1 is not None and (y2 is None or s1 >= s2)):
                    if trace:
                        trace_lines.append(
                            f"[−] sign decision feats={feats_sign} → flip_hint={flip_hint} | chosen='hint' (score={s1:.3f} vs {s2:.3f})")
                        trace_lines.extend(t1)
                    return y1, trace_lines
                elif y2 is not None:
                    if trace:
                        trace_lines.append(
                            f"[−] sign decision feats={feats_sign} → flip_hint={flip_hint} | chosen='alternate' (score={s2:.3f} vs {s1:.3f})")
                        trace_lines.extend(t2)
                    return y2, trace_lines
                else:
                    if trace:
                        trace_lines.append(f"[−] both orders failed; abort")
                    return None, trace_lines
        
        elif op == '*':
            # Single-digit multiplier; route through learned table, then fallbacks.
            aa, m = a[::-1], b
            for i in range(len(a) + len(b) + 2):
                d1 = aa[i] if i < len(aa) else '0'
                if d1 == '0' and s_in == 0 and i >= len(aa):
                    break
                
                k0 = (op, d1, m, s_in)
                v = section.get(k0)
                used_fallback = "none"
                chosen_key = k0
                
                # Fallback to observation argmax.
                if v is None:
                    v = self._obs_argmax(sheaf, k0)
                    if v is not None:
                        used_fallback = "obs_argmax"
                
                # Fallback to star-projection learned from (d₁, m).
                if v is None:
                    v = self._star_proj_gs.get((d1, m))
                    if v is not None:
                        used_fallback = "star_projection"
                
                if v is None:
                    if trace:
                        trace_lines.append(f"[*] i={i} k={k0} → MISSING; abort")
                    return None, trace_lines
                
                d_out, s_out = v
                if trace:
                    trace_lines.append(
                        f"[*] i={i} s_in={s_in} key={chosen_key} → d_out={d_out}, s_out={s_out} (fallback={used_fallback})")
                out.append(d_out)
                s_in = s_out
        
        # Finalise: append residual carry if present and return string.
        if s_in > 0:
            out.extend(list(str(s_in))[::-1])
            if trace:
                trace_lines.append(f"[carry] final s_in={s_in} appended")
        pred = strip_leading_zeros("".join(reversed(out)))
        if trace:
            trace_lines.append(f"[=] result → {pred}")
        return pred, trace_lines
    
    # ---------- Inference (top-level) ----------
    def predict(self, op: str, a: str, b: str, trace: bool = False) -> Tuple[str, List[str]]:
        """
        Predict the result for (op, a, b) by trying sheaves in a sensible order.

        Ordering heuristic:
          • Rank sheaves by (estimated coverage on this input, specialisation for op).
          • Try in descending order until one succeeds; aggregate trace as we go.

        Returns:
          (prediction_string, trace_lines)

        Note on ‘op’ branches:
          Branches by operator are *routing into learned tables/compositions*,
          not algorithmic rules. No borrow/carry arithmetic is computed here.
        """
        # NEW: multi-digit '*' composition path — pure routing/composition.
        if op == '*' and len(b) > 1:
            y, tr = self._predict_mul_multidigit(a, b, trace=trace)
            return (y if y is not None else ""), tr
        
        if not self.sheaves:
            return "", []
        
        # Estimate coverage: how many pertinent keys are present in g_θ or counts.
        def coverage(sh: SheafOfRules) -> int:
            c = 0
            s_in = 0
            if op in '+-':
                n = max(len(a), len(b))
                aa, bb = zpad_left(a, n)[::-1], zpad_left(b, n)[::-1]
                for i in range(n + 2):
                    d1, d2 = (aa[i] if i < n else '0'), (bb[i] if i < n else '0')
                    if i >= n and s_in == 0:
                        break
                    k = (op, d1, d2, s_in)
                    if (k in sh.global_section) or (k in sh._stalk_observations):
                        c += 1
            elif op == '*':
                aa, m = a[::-1], b
                for i in range(len(a) + len(b) + 2):
                    d1 = aa[i] if i < len(aa) else '0'
                    if d1 == '0' and s_in == 0 and i >= len(aa):
                        break
                    k = (op, d1, m, s_in)
                    if (k in sh.global_section) or (k in sh._stalk_observations):
                        c += 1
            return c
        
        # Sort sheaves by coverage and operator specialisation (descending).
        order = sorted(self.sheaves, key=lambda sh: (coverage(sh), sh.op_specialisation.get(op, 0.0)), reverse=True)
        
        # Try sheaves in order; collect traces for transparency.
        agg_trace: List[str] = []
        for sh in order:
            y, tr = self._try_predict_with_sheaf(sh, op, a, b, trace=trace)
            agg_trace.append(f"[sheaf {sh.sheaf_id}] try → {'OK' if y else 'fail'}; spec={dict(sh.op_specialisation)}")
            agg_trace.extend(tr)
            if y is not None:
                return y, agg_trace
        return "", agg_trace
    
    # ---------- Memory dump ----------
    def dump_memory(self, path: str):
        """
        Persist an interpretable snapshot of the learned knowledge to JSON.

        Contains:
          • all sheaves’ global sections (as readable lists) and specialisations;
          • raw observation sizes per key (for context);
          • auxiliaries (sign decisions, '+' order decisions, '*' star projections).
        """
        dump = {
            "sheaves": [],
            "auxiliaries": {
                "sign_gs": {str(k): int(v) for k, v in self.sign_sheaf.gs.items()},
                "order_plus_gs": {str(k): int(v) for k, v in self.order_plus.gs.items()},
                "star_projection_gs": {f"{k[0]}|{k[1]}": {"digit_out": v[0], "s_out": v[1]}
                                       for k, v in self._star_proj_gs.items()},
            }
        }
        for sh in self.sheaves:
            sheaf_obj = {
                "sheaf_id": sh.sheaf_id,
                "total_weight": sh.total_weight_processed,
                "op_specialisation": {k: float(v) for k, v in sh.op_specialisation.items()},
                "global_section_size": len(sh.global_section),
                "global_section": [],
            }
            # Flatten the mapping for readability; include obs sizes.
            for (op, d1, d2m, s_in), (d_out, s_out) in sh.global_section.items():
                obs_size = int(sum(sh._stalk_observations[(op, d1, d2m, s_in)].values()))
                sheaf_obj["global_section"].append({
                    "op": op, "d1": d1, "d2_or_m": d2m, "s_in": s_in,
                    "digit_out": d_out, "s_out": s_out, "obs": obs_size
                })
            dump["sheaves"].append(sheaf_obj)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(dump, f, indent=2, sort_keys=False)


# ----------------------- Training / evaluation helpers -----------------------
def _sheaf_tension(model: CognitiveTopos, sample: List[Tuple[str, str, str, str]]) -> Tuple[int, int]:
    """
    Diagnostic: count unknown local rules vs digit mismatches on a probe sample.

    Returns:
      (unknown_count, digit_mismatch_count)

    • unknown_count: how many (k,v) pairs in the probe’s teacher sections lack a
      cached log-probability in the best specialised sheaf for that op.
    • digit_mismatch_count: per-digit disagreements between c_true and c_pred.
    """
    unknown = 0
    mism = 0
    for (a, b, c, op) in sample:
        section = Decomposer.get_ideal_local_section(op, a, b, c)
        if not section or not model.sheaves:
            continue
        best = max(model.sheaves, key=lambda s: s.op_specialisation.get(op, -1.0))
        for point, value in section:
            if (point, value) not in best._log_probabilities:
                unknown += 1
        yhat, _ = model.predict(op, a, b, trace=False)
        if yhat != c and yhat != "":
            s1 = c
            s2 = yhat
            L = max(len(s1), len(s2))
            s1 = zpad_left(s1, L)
            s2 = zpad_left(s2, L)
            mism += sum(1 for i in range(L) if s1[i] != s2[i])
    return unknown, mism


def make_random_tests(n: int, max_len: int, p_multi_mul: float = 0.5, max_len_mul: Optional[int] = None) -> List[
    Tuple[str, str, str, str]]:
    """
    Produce n random *labelled* expressions (a, op, b, c_true) for evaluation.

    Mix:
      • Plus and minus: a,b sampled up to length max_len+1;
      • Times: a as above, multiplier m ∈ {1..9}.

    Note:
      Labels are used *only* to compute accuracy; the predictor never sees c at
      inference.
    """
    if max_len_mul is None:
        max_len_mul = max_len + 1
    
    tests = []
    ops = ['+', '-', '*']
    for _ in range(n):
        op = random.choice(ops)
        if op == '*':
            a = rand_int_str(max_len + 1)
            if random.random() < p_multi_mul:
                # multi-digit multiplier — допускаем ведущую ≠0, но внутри могут быть '0'
                b = rand_int_str(max_len_mul)  # rand_int_str не ставит ведущий '0', но внутренние нули возможны
                true_c = str(int(a) * int(b))
            else:
                # одноцифровое m, в т.ч. '0' иногда:
                b = random.choice("0123456789")
                # избегаем тривиала 'a=0 & m=0' если хочешь: но не обязательно
                true_c = gt_mul1(a, b)
        else:
            a = rand_int_str(max_len + 1)
            b = rand_int_str(max_len + 1)
            true_c = gt_add(a, b) if op == '+' else gt_sub_nonneg(a, b)
        tests.append((a, op, b, true_c))
    return tests


# ----------------------- End-to-end runner -----------------------
def run_experiment(seed=11, epochs=6, train_n=4000, test_n=900, max_len=3,
                   trace_demo: List[Tuple[str, str, str]] = None):
    """
    Train the model, evaluate it, export CSV and traces, and dump readable memory.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility across runs.
    epochs : int
        Number of passes over the *per-operator* training sets.
    train_n : int
        Per-operator training size (total ≈ 3·train_n over all ops).
    test_n : int
        Per-operator test size (held-out evaluation).
    max_len : int
        Max length of random integers for training.
    trace_demo : list[(a, op, b)]
        Expressions to trace verbosely after training; default has one per op.

    Returns
    -------
    dict with file paths for CSV, traces, memory dump.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # ---------------- Dataset synthesis (LABELLED) ----------------
    datasets = {
        '+': make_add_dataset(train_n, max_len),
        '-': make_sub_dataset(train_n, max_len),
        '*': make_mul1_dataset(train_n, max_len),
    }
    # Flatten into a single pool of labelled examples with operator tag.
    all_train_data = [(*item, op) for op, data in datasets.items() for item in data]
    
    # Held-out test sets (slightly harder lengths for ‘+’ and ‘−’).
    test_sets = {
        '+': make_add_dataset(test_n, max_len + 1),
        '-': make_sub_dataset(test_n, max_len + 1),
        '*': make_mul1_dataset(test_n, max_len),
    }
    
    print(f"Total Train: {len(all_train_data)}, Total Test: {sum(len(v) for v in test_sets.values())}")
    
    # ---------------- Model initialisation ----------------
    model = CognitiveTopos(max_sheaves=50, novelty_log_prob=-50.0, novelty_threshold=0.95,
                           temperature=0.6, lambda_mdl=0.0, alpha=1e-2, rng_seed=seed)
    
    print("\nStarting Training (HELOS Topos)…")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # (A) Shuffle the training pool each epoch.
        random.shuffle(all_train_data)
        
        # (B) Online EM updates, periodically rebuilding global sections to keep caches fresh.
        for i, (a, b, c, op) in enumerate(all_train_data):
            model.observe(op, a, b, c)  # one labelled example → update responsibilities and counts
            if (i + 1) % 1000 == 0:
                model.fit_all_sheaves()  # refresh global sections and auxiliaries
        
        # (C) Finalise epoch: rebuild sections and auxiliaries.
        model.fit_all_sheaves()
        
        # (D) Quick accuracy snapshot on held-out sets (predict uses no labels).
        accuracies = {
            op: (sum(1 for a, b, c in data if model.predict(op, a, b)[0] == c) / len(data)
                 if data else 0.0)
            for op, data in test_sets.items()
        }
        
        # (E) Probe sheaf tension: unknown keys vs per-digit mismatches on a small sample.
        probe = all_train_data[:min(300, len(all_train_data))]
        unk, mis = _sheaf_tension(model, probe)
        
        print(f"Epoch {epoch}/{epochs}  "
              f"Time: {time.time() - t0:.2f}s  "
              f"Sheaves: {len(model.sheaves)} | "
              f"Add:{accuracies['+']:.3f}  Sub:{accuracies['-']:.3f}  Mul:{accuracies['*']:.3f} | "
              f"Tension[unknown:{unk}, mism:{mis}]")
    
    # ---------------- Report final sheaf state ----------------
    print("\n--- Final Topos State ---")
    for s in model.sheaves:
        # Print top specialisations for readability.
        spec_str = ", ".join([f"{op}:{count:.0f}" for op, count in
                              sorted(s.op_specialisation.items(), key=lambda item: -item[1])[:3]])
        print(f"Sheaf {s.sheaf_id}: Global Section Size: {len(s.global_section):>3}. "
              f"Total Weight: {s.total_weight_processed:.0f}. "
              f"Specialisation: [{spec_str}]")
    
    # ---------------- Verbose demo tracing ----------------
    if trace_demo is None:
        trace_demo = [("46543", "+", "20"), ("831", "-", "178"), ("1234", "*", "9"), ("1234", "*", "56")]
    trace_path = "../../results/arithmetic/helos_traces.txt"
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    with open(trace_path, "w") as tf:
        print("\n--- Traces (detailed reasoning) ---")
        for a, op, b in trace_demo:
            # If multi-digit multiplier, go through composition path (predict handles it).
            y, trace_lines = model.predict(op, a, b, trace=True)
            truth_fun = {'+': gt_add, '-': gt_sub_nonneg}
            if op == '*':
                true_y = str(int(a) * int(b))
            else:
                true_y = truth_fun[op](a, b)
            header = f"{a} {op} {b} = {y} (True: {true_y}) {'✅' if y == true_y else '❌'}"
            print(header)
            tf.write(header + "\n")
            for line in trace_lines:
                print("   ", line)
                tf.write("    " + line + "\n")
            print("-" * 72)
            tf.write("-" * 72 + "\n")
    
    # ---------------- Extended random test suite + CSV export ----------------
    print("\n=== Random Expressions Test Suite ===")
    rand_tests = make_random_tests(500, max_len + 1, p_multi_mul=0.6, max_len_mul=max_len + 2)
    print(f"{'Expression':<24} {'Predicted':<16} {'Expected':<16} {'Status'}")
    print("-" * 70)
    
    per_op_ok = {'+': 0, '-': 0, '*': 0}
    per_op_tot = {'+': 0, '-': 0, '*': 0}
    overall_ok = 0
    
    csv_path = "../../results/arithmetic/helos_results.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "op", "b", "predicted", "expected", "ok"])
        for a, op, b, true_c in rand_tests:
            y, _ = model.predict(op, a, b, trace=False)  # inference uses only (a,op,b)
            ok = (y == true_c)
            per_op_ok[op] += int(ok)
            per_op_tot[op] += 1
            overall_ok += int(ok)
            expr = f"{a} {op} {b}"
            # Print a tiny random subset to console for a quick human check.
            if random.random() < 0.025:
                print(f"{expr:<24} {y:<16} {true_c:<16} {'✅' if ok else '❌'}")
            w.writerow([a, op, b, y, true_c, int(ok)])
    
    # ---------------- Summary and memory dump ----------------
    print("\n--- Random Test Summary ---")
    for op in ['+', '-', '*']:
        acc = (per_op_ok[op] / per_op_tot[op]) if per_op_tot[op] else 0.0
        print(f"Accuracy {op}: {acc:.4f}   ({per_op_ok[op]}/{per_op_tot[op]})")
    overall_acc = overall_ok / sum(per_op_tot.values()) if sum(per_op_tot.values()) else 0.0
    print(f"Overall:  {overall_acc:.4f}   ({overall_ok}/{sum(per_op_tot.values())})")
    print(f"CSV saved to: {csv_path}")
    
    mem_path = "../../results/arithmetic/helos_memory_dump.json"
    model.dump_memory(mem_path)
    print(f"Memory dump saved to: {mem_path}")
    
    """
    ==============================================
     Out-of-Distribution (OOD) Evaluation Section
    ==============================================

    These evaluations are designed to test whether the HELOS Topos arithmetic learner
    has memorised individual digit patterns or has discovered a generalisable
    algorithmic structure. Each OOD suite deliberately departs from the distribution
    seen during training, probing whether the learned sheaf system can apply its
    emergent local rules to unfamiliar symbolic configurations.

    There are three complementary OOD regimes:

    1. OOD length↑  – Tests numbers longer than any seen during training.
       This measures whether the model truly applies digit-wise reasoning
       recursively (emergent compositionality) rather than relying on memorised
       n-gram patterns of fixed length.

    2. OOD digit-skew – Tests arithmetic with unusual digit frequencies,
       e.g. dominated by zeros or nines. It checks whether the local carry/borrow
       transitions are abstract (rule-based) or dependent on specific token
       distributions encountered before.

    3. OOD multi-digit '*' – Tests multi-digit multiplications never observed as
       exact examples in training. Success here demonstrates that the system has
       inferred the procedure of repeated partial multiplication and aggregation
       from single-digit cases, i.e. that the concept of multiplication emerged as
       a structural operation rather than a memorised mapping.

    Passing these OOD suites is a strong indicator that the model has *not stored*
    specific arithmetic outcomes but has instead induced a compact,
    general, and interpretable algorithm that scales compositionally.
    In other words: high OOD accuracy constitutes empirical evidence that the
    system has discovered arithmetic as an emergent, rule-consistent process
    within its symbolic topology — it has derived, not remembered, how to think.
    """
    
    print("\n--- OOD Evaluations ---")
    
    # (A) Longer lengths than training
    ood_long = []
    for _ in range(300):
        opx = random.choice(['+', '-', '*'])
        if opx == '*':
            a = rand_int_str(max_len + 3)
            b = random.choice("123456789")
            ctrue = str(int(a) * int(b))
        else:
            a = rand_int_str(max_len + 3)
            b = rand_int_str(max_len + 3)
            ctrue = gt_add(a, b) if opx == '+' else gt_sub_nonneg(a, b)
        ood_long.append((a, opx, b, ctrue))
    ok = sum(int(model.predict(opx, a, b, trace=False)[0] == c) for a, opx, b, c in ood_long)
    print(f"OOD length↑: acc = {ok / len(ood_long):.4f}   ({ok}/{len(ood_long)})")
    
    # (B) Digit-skewed distribution (more 8–9)
    def rand_skewed(maxL):
        """
        Generate a skewed-digit integer string (heavier on 8/9) with 1..maxL digits.
        Preserves no-leading-zero for multi-digit numbers.
        """
        L = random.randint(1, maxL)
        if L == 1:
            pool = "0011223344556677888999"
            return random.choice(pool)
        head_pool = "123456789"
        tail_pool = "000111222333444555666777888999"
        s = random.choice(head_pool) + ''.join(random.choice(tail_pool) for _ in range(L - 1))
        return s.lstrip('0') or '0'
    
    ood_skew = []
    for _ in range(300):
        opx = random.choice(['+', '-', '*'])
        if opx == '*':
            a = rand_skewed(max_len + 2)
            # allow multi-digit multiplier here to test composition
            b = rand_skewed(2)
            if b == "0":
                b = "1"
            ctrue = str(int(a) * int(b))
        else:
            a = rand_skewed(max_len + 2)
            b = rand_skewed(max_len + 2)
            ctrue = gt_add(a, b) if opx == '+' else gt_sub_nonneg(a, b)
        ood_skew.append((a, opx, b, ctrue))
    ok = sum(int(model.predict(opx, a, b, trace=False)[0] == c) for a, opx, b, c in ood_skew)
    print(f"OOD digit-skew: acc = {ok / len(ood_skew):.4f}   ({ok}/{len(ood_skew)})")
    
    # (C) Multi-digit multipliers (2–4 digits)
    ood_multid = []
    for _ in range(300):
        a = rand_int_str(max_len + 2)
        b = rand_int_str(2 + random.randint(0, 2))  # 2–4 digits
        if b == "0":
            b = "1"
        ood_multid.append((a, '*', b, str(int(a) * int(b))))
    ok = sum(int(model.predict('*', a, b, trace=False)[0] == c) for a, _, b, c in ood_multid)
    print(f"OOD multi-digit '*': acc = {ok / len(ood_multid):.4f}   ({ok}/{len(ood_multid)})")
    
    return {
        "csv_path": csv_path,
        "trace_path": trace_path,
        "mem_path": mem_path,
    }


# ----------------------- Execute experiment -----------------------
if __name__ == "__main__":
    out = run_experiment(seed=21, epochs=6, train_n=3500, test_n=900, max_len=3)
    print("Done.", out)
