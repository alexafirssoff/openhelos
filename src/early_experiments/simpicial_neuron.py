import math
import numpy as np
import random
import matplotlib.pyplot as plt
import logging
import traceback

from pathlib import Path
from typing import (
    Tuple,
    Any,
    Optional
)
from gudhi.simplex_tree import SimplexTree
from collections import defaultdict
from binstorage import BinaryStorage
from pairing_function import pair
from dataclasses import dataclass
from familiarity_score import calculate_familiarity_score

# Basic logger setup
logging.basicConfig(
    level=logging.INFO,  # Set the minimum level for processing
    format='%(asctime)s - %(levelname)s - %(message)s'  # Output format
)

# Create logger
neuron_logger = logging.getLogger('Neuron')

# Configure handler (output to console)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


@dataclass
class NeuronHyperParams:
    """
    Hyperparameters controlling the behaviour and learning of a SimplicialNeuron.
    """
    
    # --- Metric calculation parameters ---
    
    epsilon: float = 1e-10
    """A small positive constant to avoid logarithm of zero
       (e.g., in `compute_free_energy` when `P_pair = 0`)."""
    
    freq_damping_factor: float = 10.0
    """Frequency damping factor (lambda_f) in the Free Energy (F) formula.
       Determines how quickly the pair frequency `f_t` suppresses
       the "local surprise" term (related to P_pair).
       A LARGER value -> frequency has a SLOWER/WEAKER effect."""
    
    alpha_p_single: float = 0.2
    """Weighting coefficient (alpha_p) for the frequency component (freq_score)
       in the Element Stability (P_single) formula. Range [0, 1].
       A value of 0.7 means that 70% of P_single is determined by frequency,
       and 30% by topological centrality."""
    
    k_familiarity: float = 0.7
    """Coefficient 'k' in the exponential function for calculating the Familiarity Score (FS)
       from Free Energy (F): FS = exp(-k * F). Determines the steepness of the transformation:
       A LARGER 'k' value -> FS drops faster as F increases (higher sensitivity)."""
    
    # --- History and plasticity parameters ---
    
    max_history_s: int = 1000
    """Maximum history length (`N_hist`) used for calculating
       expected stability (`S_expected`) and, potentially, for F-based plasticity."""
    
    default_plasticity_control: str = 'S_based'
    """Default mode for controlling the neuron's internal plasticity.
       Possible values:
       - 'S_based': Plasticity `p_local = exp(-gamma_S * S)`. Decreases as stability S increases.
       - 'F_based': Plasticity `p_local = 1 - FS_avg`. Decreases as average familiarity FS increases.
       - 'none': Plasticity is always 1.0 (neuron is always maximally open to changes)."""
    
    default_plasticity_decay_rate: float = 5.0
    """Decay parameter (`gamma_S` or `gamma_F`) for 'S_based' or 'F_based' plasticity control modes.
       Determines how quickly plasticity drops as S or FS increases."""
    
    min_modulation_for_history: float = 0.1
    """Minimum value of the external modulation signal `m` at which
       the neuron's metric history and weight training (if any) occur.
       Allows separation of 'train' (m ~ 1.0) and 'predict' (m ~ 0.0) modes."""
    
    # --- P_single "maturation" parameters ---
    
    p_single_maturity_start: int = 1
    """Minimum number of vertices (`N_v^start`) from which
       the maturity factor `maturity` for `P_single` starts to become non-zero."""
    
    p_single_maturity_ramp: float = 3.0  # Using float for greater flexibility in the formula
    """Length of the "ramp" (`N_v^ramp`) for the `P_single` maturity factor.
       The factor reaches 1.0 when the number of vertices reaches
       `N_v^start + N_v^ramp` (e.g., 2 + 3 = 5 vertices).
       Formula: maturity = max(0, min(1, (Nv - Nv_start) / Nv_ramp))."""
    
    # --- Edge weight parameters (if used) ---
    w_init: float = 0.1  # Initial weight of a new edge
    eta_w: float = 0.01  # Weight learning rate (Hebbian)
    gamma_w: float = 0.001  # Weight decay rate
    lambda_w: float = 1.0  # Coefficient of weight influence on F (in exp(-w/lambda_w))


class SimplicialNeuron:
    """
    Implements the Simplicial Neuron (SN) model - an adaptive computational element
    inspired by principles of structural plasticity and self-organisation.

    Unlike traditional neurons, an SN represents internal knowledge
    as a dynamically changing combinatorial simplicial complex (graph),
    where vertices correspond to unique input symbols (represented
    by global IDs), and edges correspond to observed binary relationships (pairs).

    Learning occurs through structural plasticity: stochastic addition
    of vertices for new symbols (via an external `get_global_vertex_id`) and deterministic
    addition/strengthening of edges for observed pairs (internal `complex` and `freq`).
    The neuron's plasticity is modulated by its internal state (stability S or
    free energy F) and an external signal.

    When a pair of input vertex IDs (v1, v2) is presented, the neuron computes a set
    of local metrics characterising its response and the state of its internal
    complex:
    - S: Global stability (density) of the neuron's local graph.
    - P_pair: Local stability of the presented pair (structural support).
    - P_single: Local stability/centrality of individual vertices.
    - F: Free energy (a measure of "surprise") for the presented pair.
    - b0: Number of connected components of the neuron's local graph.

    The neuron also generates an output identifier 'z' (based on `pair(v1, v2)`,
    possibly order-invariant), which represents the processed pair
    and can serve as input for the next level of hierarchical processing.

    The SN is intended for use as a building block in hierarchical
    architectures (e.g., `Column`) for tasks involving unsupervised streaming analysis
    of sequence structure.
    """
    
    def __init__(
            self,
            name: int,
            storage_path: Path,
            hyperparameters=NeuronHyperParams,
            ordered_z=False
    ):
        """
        Initialises or loads the state of a simplicial neuron.

        Creates a neuron with a unique `name` and a path for data storage
        `storage_path`. If a state file for this neuron already exists,
        its state is loaded (including topology, frequencies, histories, and parameters).
        Otherwise, the neuron is initialised with starting values, using
        the provided hyperparameters or default values.

        Args:
            name: Unique integer identifier for the neuron.
            storage_path: Path object to the directory where neuron state files are stored.
                          The file for this neuron will be `{storage_path}/{name}.bin`.
        """
        
        self.name = name
        self.storage_path = storage_path
        self.hyperparameters = hyperparameters
        self.ordered_z = ordered_z
        
        full_path = self.storage_path / f'{self.name}.bin'
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # === Random Number Generator for Heterogeneity ===
        # Use the neuron's name as a seed for the local generator.
        # This ensures that DIFFERENT neurons receive DIFFERENT initial variations,
        # but THE SAME neuron upon restart (if the state does not load)
        # will receive THE SAME variations.
        self._local_random = random.Random(self.name)
        
        # -----------------------------------------------------
        
        # --- Function for Random Initial Weight ---
        # This function will be unique for each neuron instance
        # due to the use of self._local_random
        def _random_w_init():
            w_init_base = self.hyperparameters.w_init
            # Define the noise range (e.g., +/- 10% of the base)
            w_init_noise_range = w_init_base * 0.4
            noise = self._local_random.uniform(-w_init_noise_range / 2, w_init_noise_range / 2)
            # Return the base weight + noise, but not less than zero
            return max(0.0, w_init_base + noise)
        
        # --------------------------------------------
        
        # Initial memory state
        initial_memory = {
            'complex': SimplexTree(),
            'freq': defaultdict(int),
            'max_freq': 0,
            'max_history_len': 1000,
            'S_history': [],
            'F_history': [],
            'P_pair_history': [],
            'b0_history': [],
            'edge_weights': defaultdict(_random_w_init)
        }
        
        # Load or initialise storage
        try:
            self.memory = BinaryStorage(full_path, initial_memory)
        except Exception as e:
            neuron_logger.critical(f"CRITICAL: Error init/load N {self.name} storage: {e}. Attempting to reset.")
            traceback.print_exc()
            if full_path.exists():
                try:
                    full_path.unlink()
                except OSError as e_unlink:
                    neuron_logger.critical(f"CRITICAL: Failed to delete corrupted file {full_path}: {e_unlink}")
                    raise  # Re-raise if we cannot even delete
            
            # Retry initialisation from scratch
            self.memory = BinaryStorage(full_path, initial_memory)
        
        self.complex: SimplexTree = self.memory.data.get('complex', initial_memory['complex'])
        if not isinstance(self.complex, SimplexTree):
            neuron_logger.warning(f"WARNING N {self.name}: Invalid complex type loaded, resetting.")
            self.complex: SimplexTree = initial_memory['complex']
        
        loaded_freq = self.memory.data.get('freq', initial_memory['freq'])
        # Convert to defaultdict(int) if it is a regular dict
        if isinstance(loaded_freq, dict) and not isinstance(loaded_freq, defaultdict):
            self.freq = defaultdict(int, loaded_freq)
        elif isinstance(loaded_freq, defaultdict):
            self.freq = loaded_freq
        else:
            neuron_logger.warning(f"WARNING N {self.name}: Invalid freq type loaded, resetting.")
            self.freq = initial_memory['freq']
        
        self.max_history_len = self.memory.data.get('max_history_len', self.hyperparameters.max_history_s)
        self.max_freq = self.memory.data.get('max_freq', initial_memory['max_freq'])
        
        # Histories - simply load lists
        self.S_history = self.memory.data.get('S_history', initial_memory['S_history'])
        self.F_history = self.memory.data.get('F_history', initial_memory['F_history'])
        self.P_pair_history = self.memory.data.get('P_pair_history', initial_memory['P_pair_history'])
        self.b0_history = self.memory.data.get('b0_history', initial_memory['b0_history'])
        
        # Additional type checks for histories (must be lists)
        for history_name in ['S_history', 'F_history', 'P_pair_history', 'b0_history']:
            attr_value = getattr(self, history_name)
            if not isinstance(attr_value, list):
                print(f"Warning N {self.name}: Invalid {history_name} type loaded ({type(attr_value)}), resetting.")
                setattr(self, history_name, [])
        
        # # --- Add weights ---
        # w_init = hyperparameters.w_init
        # initial_memory['edge_weights'] = defaultdict(
        #     lambda: _random_w_init)  # Default weight upon first access
        
        # Loading Edge Weights
        loaded_weights_dict = self.memory.data.get('edge_weights', {})  # Load as dict
        weights_with_frozenset_keys = {}
        if isinstance(loaded_weights_dict, dict):  # Check that it is a dictionary
            for k_iterable, v in loaded_weights_dict.items():
                try:
                    # Keys might have been saved as lists/tuples
                    key_frozen = frozenset(k_iterable)
                    weights_with_frozenset_keys[key_frozen] = float(v)
                except (TypeError, ValueError):
                    neuron_logger.warning(
                        f"N {self.name}: Skipping invalid edge weight key during load: {k_iterable}")
        else:
            neuron_logger.warning(f"N {self.name}: Loaded edge_weights is not a dict, resetting weights.")
        
        # Create defaultdict with OUR RANDOM factory and populate with loaded ones
        self.edge_weights = defaultdict(_random_w_init, weights_with_frozenset_keys)
        # Explicitly set the factory, as it is not saved/loaded by pickle
        self.edge_weights.default_factory = _random_w_init
        
        # # Loading weights
        # loaded_weights = self.memory.data.get('edge_weights', initial_memory['edge_weights'])
        # if isinstance(loaded_weights, dict) and not isinstance(loaded_weights, defaultdict):
        #     # Important: keys - frozenset, values - float
        #     # Convert keys back to frozenset if they were saved as tuples/lists
        #     weights_with_frozenset_keys = {}
        #     for k, v in loaded_weights.items():
        #         try:
        #             # Attempt to create frozenset if the key is an iterable object (tuple, list)
        #             key_frozen = frozenset(k)
        #             weights_with_frozenset_keys[key_frozen] = float(v)
        #         except (TypeError, ValueError):
        #             neuron_logger.warning(f"N {self.name}: Skipping invalid edge weight key during load: {k}")
        #     w_init = self.hyperparameters.w_init
        #     self.edge_weights = defaultdict(lambda: w_init, weights_with_frozenset_keys)
        #
        # elif isinstance(loaded_weights, defaultdict):
        #     self.edge_weights = loaded_weights
        #     # Set the default factory in case it was not saved
        #     w_init = self.hyperparameters.w_init
        #     self.edge_weights.default_factory = lambda: w_init
        # else:
        #     neuron_logger.warning(f"WARNING N {self.name}: Invalid edge_weights type loaded, resetting.")
        #     self.edge_weights = initial_memory['edge_weights']
        # # --------------------
    
    def _get_local_plasticity_level(self) -> float:
        """
        Calculates the internal (local) plasticity level of the neuron.

        The plasticity level determines the probability of adding a new, previously
        unknown vertex (symbol) to the neuron's simplicial complex.
        The calculation depends on the selected `self.plasticity_control` mode:
          - 'none': Plasticity is always maximal (1.0).
          - 'S_based': Plasticity decreases exponentially as global
                       stability S increases: exp(-gamma * S).
          - 'F_based': Plasticity is inversely proportional to familiarity, estimated
                       from the moving average of free energy F: 1 - FS(avg(F)).

        Returns:
            Local plasticity value in the range [0.0, 1.0].
        """
        if self.hyperparameters.default_plasticity_control == 'none':
            return 1.0
        elif self.hyperparameters.default_plasticity_control == 'S_based':
            # Use try-except for safe access to history
            try:
                current_S = self.S_history[-1] if self.S_history else 0.0
            except IndexError:
                current_S = 0.0
            plasticity = math.exp(-self.hyperparameters.default_plasticity_decay_rate * current_S)
            return max(0.0, min(1.0, plasticity))  # Limit to [0, 1]
        
        elif self.hyperparameters.default_plasticity_control == 'F_based':
            if not self.F_history:
                return 1.0
            try:
                window_size = min(len(self.F_history), 10)
                avg_F = np.mean(self.F_history[-window_size:])
                
                # Pass k_familiarity from neuron attributes
                plasticity = 1.0 - calculate_familiarity_score(avg_F, k=self.hyperparameters.k_familiarity)
                return max(0.0, min(1.0, plasticity))
            except (IndexError, ValueError, TypeError):  # Handling mean errors
                return 1.0  # Default behaviour on error
        return 1.0  # Default if control type is unknown
    
    def _touch_vertex(self, symbol: Any) -> int:
        """
        Converts an input symbol to a complex vertex ID.

        If the symbol is already known to the neuron (present in `self.complex`),
        it returns the symbol itself.

        Args:
            symbol: Input symbol (any hashable type).

        Returns:
            Vertex ID (int >= 0), if the symbol is known or was successfully added.
        """
        
        vertices = set(v for simplex in self.complex.get_skeleton(0) for v in simplex[0])
        if symbol in vertices:
            # column_manager_logger.info(f'INFO N {self.name}: Vertex "{symbol}" already presented in complex')
            return symbol
        
        try:
            # Add vertex (0-simplex)
            self.complex.insert([symbol], filtration=0.0)
            return symbol
        except Exception as e:
            # Log the error, but continue (vertex added to complex)
            neuron_logger.warning(f"Warning N {self.name}: Error inserting vertex {symbol} into complex: {e}")
        
        return -1
    
    def _touch_pair(self, x1: Any, x2: Any) -> Tuple[int, int]:
        """
        Processes the presentation of a pair (x1, x2), updating the neuron's state.

        1. Increments the frequency of the pair `(x1, x2)` in `self.freq` and updates `self.max_freq`.
        2. Calculates effective plasticity based on local plasticity and `modulation_signal`.
        3. Calls `_touch_vertex` for `x1` and `x2` to obtain/create vertices `v1`, `v2`.
        4. If both vertices `v1`, `v2` are successfully obtained/created (not equal to -1):
           - Adds or updates the edge `(v1, v2)` in `self.complex`, using
             the current pair frequency `f_t(x1, x2)` as the filtration value.

        Args:
            x1: First symbol of the pair.
            x2: Second symbol of the pair.

        Returns:
            Tuple `(v1, v2)` with vertex IDs if the pair was successfully processed.
            Tuple `(-1, -1)` if processing was interrupted due to ignoring
            one of the symbols at low plasticity.
        """
        
        pair_key_freq = (x1, x2)  # Order might be important for frequencies
        self.freq[pair_key_freq] += 1
        self.max_freq = max(self.max_freq, self.freq[pair_key_freq])
        
        v1 = self._touch_vertex(x1)
        v2 = self._touch_vertex(x2)
        
        if v1 == -1 or v2 == -1:
            return -1, -1
        
        # Key for weights is unordered
        edge_key_weight = frozenset({v1, v2})
        
        try:
            is_new_edge = not self.complex.find([v1, v2])  # Check if the edge is new BEFORE insertion
            # Insert/update edge in the complex
            self.complex.insert([v1, v2], filtration=float(self.freq[pair_key_freq]))
            # Initialise weight if the edge is new (or if it's not in the dictionary - default_factory will trigger)
            # Accessing by key will call default_factory if the key is not present
            _ = self.edge_weights[edge_key_weight]  # This initialises weight with w_init if the edge was not present
            # if is_new_edge:
            #    column_manager_logger.debug(f"N {self.name}: Initialized weight for new edge {edge_key_weight} to {self.hyperparameters.w_init}")
        except Exception as e:
            neuron_logger.warning(f"Warning N {self.name}: Error inserting edge ({v1},{v2}): {e}")
        
        return v1, v2
    
    def _update_edge_weight(self, v1: int, v2: int, F: float, modulation_signal: float):
        """
        Updates the weight of edge (v1, v2) based on the calculated Free Energy F.

        Uses the rule: Δw = η * FS(F) - γ * w
        where FS(F) is familiarity, η is the learning rate, and γ is decay.

        Args:
            v1: ID of the first vertex.
            v2: ID of the second vertex.
            F: Free energy calculated for this pair.
        """
        if v1 == -1 or v2 == -1 or F == float('inf') or math.isnan(F):
            return  # Do not update weight for invalid pairs or F errors
        
        edge_key = frozenset({v1, v2})
        current_weight = self.edge_weights[edge_key]  # Get current weight (or w_init)
        
        # Calculate familiarity
        familiarity = calculate_familiarity_score(F, k=self.hyperparameters.k_familiarity)
        
        effective_eta_w = self.hyperparameters.eta_w * modulation_signal
        
        gamma = self.hyperparameters.gamma_w  # Do we not modulate decay yet? Or also?
        # It is more logical to modulate only learning (eta)
        
        delta_w = effective_eta_w * familiarity - gamma * current_weight
        
        # Update rule
        # delta_w = self.hyperparameters.eta_w * familiarity - self.hyperparameters.gamma_w * current_weight
        
        # Update weight, limiting from below by zero
        new_weight = max(0.0, current_weight + delta_w)
        self.edge_weights[edge_key] = new_weight
        # === DEBUG PRINT ===
        # print(f"DEBUG TRAIN: Neuron={self.name}, Edge={{{v1},{v2}}}, "
        #       f"F={F:.4f}, FS={familiarity:.4f}, "
        #       f"CurrentW={current_weight:.4f}, DeltaW={delta_w:.4f}, NewW={new_weight:.4f}")
        # ========================
        # column_manager_logger.debug(f"N {self.name}: Updated weight for edge {edge_key}: {current_weight:.3f} -> {new_weight:.3f} (F={F:.3f}, FS={familiarity:.3f}, delta={delta_w:.4f})")
    
    def compute_stability(self) -> float:
        """
        Calculates the global stability (S) of the complex as edge density.

        S is defined as the ratio of the number of existing edges to the maximally
        possible number of edges in a graph with the current number of vertices `|V_t|`.
        A correction is applied: S is forced to 0.0 if
        `|V_t| < 3`, to avoid an uninformative S=1.0 value for
        a graph with two vertices. Updates `self.S_history`.

        Returns:
            Global stability value S in the range [0.0, 1.0].
        """
        
        num_vertices = self.complex.num_vertices()
        
        # --- CORRECTION FOR SMALL GRAPHS ---
        if num_vertices < 3:
            S = 0.0
        else:
            # -----------------------------------
            # Number of edges = (number of all simplices) - (number of vertices)
            # This is true if the complex contains only 0- and 1-simplices,
            # which is usually the case when using insert only for vertices and edges.
            num_edges = self.complex.num_simplices() - num_vertices
            # Maximum possible number of edges in a graph with N vertices
            max_edges = num_vertices * (num_vertices - 1) / 2
            S = float(num_edges / max_edges) if max_edges > 0 else 0.0  # Protection against division by zero
        
        # History update (always occurs on call)
        # It is not optimal to call append/pop each time; collections.deque can be used
        self.S_history.append(S)
        if len(self.S_history) > self.max_history_len:
            # Remove the oldest element if the limit is exceeded
            del self.S_history[0]
            # deque does this automatically when adding with maxlen
        
        return S
    
    def pair_stability(self, x1: Any, x2: Any) -> float:
        """ Calculates WEIGHTED topological stability (P_pair_w). """
        # print(f"\n--- DEBUG: pair_stability (weighted) for ('{x1}', '{x2}') ---")
        v1, v2 = x1, x2
        vertices = set(v for simplex in self.complex.get_skeleton(0) for v in simplex[0])
        
        if v1 not in vertices or v2 not in vertices:
            return 0.0
        try:
            if not self.complex.find([v1, v2]):
                return 0.0
        except Exception:
            return 0.0
        
        weighted_common_strength = 0.0
        total_weighted_degree_v1 = 0.0
        total_weighted_degree_v2 = 0.0
        neighbors_v1 = set()
        neighbors_v2 = set()
        
        try:
            # Neighbours and weighted degree of v1
            for s_tuple in self.complex.get_star([v1]):
                s = s_tuple[0]
                if len(s) == 2:
                    n = s[1] if s[0] == v1 else s[0]
                    neighbors_v1.add(n)
                    weight = self.edge_weights.get(frozenset({v1, n}),
                                                   0.0)  # Use 0.0 if weight is not present (should not happen)
                    total_weighted_degree_v1 += weight
            
            # Neighbours and weighted degree of v2
            for s_tuple in self.complex.get_star([v2]):
                s = s_tuple[0]
                if len(s) == 2:
                    n = s[1] if s[0] == v2 else s[0]
                    neighbors_v2.add(n)
                    weight = self.edge_weights.get(frozenset({v2, n}), 0.0)
                    total_weighted_degree_v2 += weight
            
            # Sum of weighted common neighbours
            common_neighbors = neighbors_v1 & neighbors_v2
            for n in common_neighbors:
                w1n = self.edge_weights.get(frozenset({v1, n}), 0.0)
                w2n = self.edge_weights.get(frozenset({v2, n}), 0.0)
                # Use minimum or product? Minimum seems more robust.
                weighted_common_strength += min(w1n, w2n)
        
        except Exception as e:
            neuron_logger.warning(f"N {self.name}: Error in weighted pair_stability for ({v1},{v2}): {e}")
            return 0.0
        
        min_weighted_degree = min(total_weighted_degree_v1, total_weighted_degree_v2)
        
        p_pair_w = float(
            weighted_common_strength / min_weighted_degree) if min_weighted_degree > self.hyperparameters.epsilon else 0.0
        # print(f"  Weighted P_pair = {weighted_common_strength:.3f} / {min_weighted_degree:.3f} = {p_pair_w:.4f}")
        # print(f"--- END DEBUG: pair_stability (weighted) ---")
        return max(0.0, min(1.0, p_pair_w))  # Limit to [0, 1]
    
    def single_stability(self, x1: Any) -> float:
        """ Calculates WEIGHTED element stability (P_single_w). """
        # print(f"\n--- DEBUG: single_stability (weighted) for '{x1}' ---")
        alpha = self.hyperparameters.alpha_p_single
        v1 = x1
        vertices = set(v for simplex in self.complex.get_skeleton(0) for v in simplex[0])
        
        if v1 not in vertices:
            return 0.0
        
        # --- Weighted degree of v1 ---
        weighted_degree_v1 = 0.0
        degree_v1 = 0  # Is the regular degree also needed for some heuristics? Not yet.
        try:
            neighbors_v1 = set()
            for s_tuple in self.complex.get_star([v1]):
                s = s_tuple[0]
                if len(s) == 2:
                    n = s[1] if s[0] == v1 else s[0]
                    neighbors_v1.add(n)
                    weight = self.edge_weights.get(frozenset({v1, n}), 0.0)
                    weighted_degree_v1 += weight
            degree_v1 = len(neighbors_v1)  # Save the regular degree
        except Exception:
            weighted_degree_v1 = 0.0
            degree_v1 = 0
        # print(f"  Weighted Degree(v1={v1}): {weighted_degree_v1:.3f} (Raw Degree: {degree_v1})")
        
        # --- Maximum WEIGHTED degree ---
        # Keep the loop, but replace with weighted degree.
        max_weighted_degree = 0.0
        all_vertices = vertices
        if not all_vertices:
            max_weighted_degree = 0.0
        else:
            for v_id in all_vertices:
                current_w_deg_v = 0.0
                try:
                    for s_tuple in self.complex.get_star([v_id]):
                        s = s_tuple[0]
                        if len(s) == 2:
                            n = s[1] if s[0] == v_id else s[0]
                            weight = self.edge_weights.get(frozenset({v_id, n}), 0.0)
                            current_w_deg_v += weight
                    max_weighted_degree = max(max_weighted_degree, current_w_deg_v)
                except Exception:
                    continue
        # print(f"  Max Weighted Degree: {max_weighted_degree:.3f}")
        
        # --- Topological score (weighted) ---
        max_w_deg_safe = max(self.hyperparameters.epsilon, max_weighted_degree)  # Use epsilon instead of 1
        # topo_score = float(weighted_degree_v1 / max_w_deg_safe) if max_w_deg_safe > 0 else 0.0
        topo_score = min(1.0, weighted_degree_v1 / 5.0)
        # print(f"  Weighted Topo Score: {topo_score:.4f}")
        
        # --- Frequency score (can be left as is, or weighted) ---
        # Option 1: Without weighting (as before)
        freq_sum = 0
        symbols_in_map = vertices
        for x_other in symbols_in_map:
            if x1 == x_other:
                continue
            freq_sum += self.freq.get((x1, x_other), 0)
            freq_sum += self.freq.get((x_other, x1), 0)
        max_freq_safe = max(1, self.max_freq) if self.max_freq > 0 else 0
        freq_score = float(freq_sum / max_freq_safe) if max_freq_safe > 0 else 0.0
        P_single_raw = alpha * freq_score + (1 - alpha) * topo_score  # topo_score DEPENDS on weights
        P_single_raw = min(1.0, max(0.0, P_single_raw))
        
        # --- Smoothing / Maturity Factor ---
        base_score_if_exists = 0.05
        P_single_modified = max(base_score_if_exists, P_single_raw)
        
        num_vertices = self.complex.num_vertices()
        maturity_factor = max(0.0, min(1.0, (
                num_vertices - self.hyperparameters.p_single_maturity_start) / self.hyperparameters.p_single_maturity_ramp))
        
        # num_vertices = self.complex.num_vertices()
        # maturity_factor = max(0.0, min(1.0, (
        #             num_vertices - self.hyperparameters.p_single_maturity_start) / self.hyperparameters.p_single_maturity_ramp))
        # P_single_final = P_single_modified * maturity_factor
        # P_single_final = max(0.0, min(1.0, P_single_final))
        base_p_single = 0.01  # New very small hyperparameter for "initial stability"
        P_single_final = base_p_single * (1.0 - maturity_factor) + P_single_raw * maturity_factor
        P_single_final = max(0.0, min(1.0, P_single_final))  # Limit to [0, 1]
        
        return P_single_final
    
    def compute_free_energy(self, x1: Any, x2: Any, P_pair: float, S: float, S_expected: float) -> float:
        """ Calculates Free Energy F, using P_pair (already weighted). """
        # FORMULA REMAINS THE SAME, but P_pair is now weighted
        pair_key = (x1, x2) if self.ordered_z else tuple(sorted([x1, x2]))
        freq_factor = float(self.freq.get(pair_key, 0))
        term1 = abs(S - S_expected)
        P_pair_safe = max(0.0, min(1.0, P_pair))
        term2_local_dissimilarity = 1.0 - P_pair_safe
        exp_arg = -freq_factor / self.hyperparameters.freq_damping_factor
        term3_freq_modulator = math.exp(exp_arg)
        F = term1 + term2_local_dissimilarity * term3_freq_modulator
        
        return F
    
    def compute_b0(self) -> int:
        """
        Calculates the number of connected components (b0, the zeroth Betti number) of the complex.

        Uses the Gudhi library to compute persistent homology
        and extract b0. In case of a Gudhi error, it uses graph traversal (BFS)
        to count connected components as a fallback mechanism.

        Returns:
            Number of connected components (int >= 0).
        """
        
        num_vertices = self.complex.num_vertices()
        if num_vertices == 0:
            return 0
        
        # Attempt to use Gudhi
        try:
            # Perhaps compute_persistence() should only be called if the complex has changed?
            # But how to track this reliably? For now, always call.
            self.complex.compute_persistence()
            betti = self.complex.betti_numbers()
            # If betti is empty or not long enough, but there are vertices, count 1 component.
            return betti[0] if betti and len(betti) > 0 else 1
        except Exception as e:
            visited = set()
            count = 0
            all_vertices_ids = list(v for simplex in self.complex.get_skeleton(0) for v in simplex[0])
            
            vertex_to_process_idx = 0
            while vertex_to_process_idx < len(all_vertices_ids):
                # Find the first unvisited vertex to start a new traversal
                start_node = -1
                while vertex_to_process_idx < len(all_vertices_ids):
                    candidate_node = all_vertices_ids[vertex_to_process_idx]
                    if candidate_node not in visited:
                        start_node = candidate_node
                        break
                    vertex_to_process_idx += 1
                
                if start_node == -1:
                    break  # All vertices visited
                
                # Start traversal of a new component
                count += 1
                q = [start_node]
                visited.add(start_node)
                head = 0
                while head < len(q):
                    curr_node = q[head]
                    head += 1
                    try:
                        neighbors = set(
                            s[0][1] if s[0][0] == curr_node else s[0][0] for s in self.complex.get_star([curr_node]) if
                            len(s[0]) == 2)
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                q.append(neighbor)
                    except Exception:
                        # Ignore errors when getting neighbours for a specific vertex
                        continue
            # If the graph is not empty, there must be at least 1 component
            return count if count > 0 else (1 if num_vertices > 0 else 0)
    
    def _trim_history(self):
        """Trims the history lists to max_history_len."""
        # Use slices to remove old elements
        for history_attr in ['S_history', 'F_history', 'P_pair_history', 'b0_history']:
            history_list = getattr(self, history_attr)
            if len(history_list) > self.max_history_len:
                setattr(self, history_attr, history_list[-self.max_history_len:])
    
    def ascending_activation(self, x1: Any, x2: Any, modulation_signal: float = 1.0) -> Tuple[
        Optional[float], Optional[float], Optional[float], Optional[int], Optional[int], Optional[float], Optional[
            float]]:
        """ Full cycle: update, calculate metrics (with weights), update weights, history, return. """
        v1, v2, z_out = -1, -1, -1  # Initialisation
        S_out, P_pair_out, F_out, b0_out = 0.0, 0.0, float('inf'), 0
        P1_out, P2_out = 0.0, 0.0
        
        # --- Step 1: Structural Update (only during training, m > min_hist) ---
        updated = False
        if modulation_signal >= self.hyperparameters.min_modulation_for_history:
            v1_upd, v2_upd = self._touch_pair(x1, x2)  # Updates frequency, complex, initialises weight
            if v1_upd != -1 and v2_upd != -1:
                v1, v2 = v1_upd, v2_upd  # Store vertex IDs for weight update
                z_out = pair(v1, v2)  # Calculate z only if the edge was added/updated
                updated = True
                # column_manager_logger.debug(f"N {self.name}: Touched pair ({x1},{x2}) -> ({v1},{v2}), z={z_out}")
        
        # --- Step 2: Metric Calculation (always, on current state) ---
        try:
            current_v1 = x1  # ID matches the symbol/value
            current_v2 = x2
            vertices = set(v for simplex in self.complex.get_skeleton(0) for v in simplex[0])
            
            # Calculate S and S_exp
            S_out = self.compute_stability()
            S_exp_out = np.mean(self.S_history) if self.S_history else S_out  # Use history BEFORE current S
            
            # Calculate P_pair_w and F only if both vertices are known
            if current_v1 in vertices and current_v2 in vertices:
                P_pair_out = self.pair_stability(current_v1, current_v2)  # P_pair is now weighted
                F_out = self.compute_free_energy(x1, x2, P_pair_out, S_out, S_exp_out)  # F uses weighted P_pair
            else:
                P_pair_out = 0.0
                F_out = float('inf')  # If no vertices, energy is infinite
            
            b0_out = self.compute_b0()
            P1_out = self.single_stability(x1)  # P_single is now weighted
            P2_out = self.single_stability(x2)
        
        except Exception as e_metrics:
            neuron_logger.error(f"N {self.name}: Error calculating metrics for ({x1}, {x2}): {e_metrics}")
            traceback.print_exc()
            # Return None to signal an error
            return None, None, None, None, None, None, None
        
        # --- Step 3: Weight Update (only during training and if pair is valid) ---
        if updated:  # If there was a structural update (i.e., m was > threshold and v1,v2 != -1)
            self._update_edge_weight(v1, v2, F_out, modulation_signal)
        
        # --- Step 4: History Update (only during training) ---
        if updated:  # Use the same flag
            try:
                # Record ONLY the just-calculated values
                self.S_history.append(S_out)
                self.F_history.append(F_out)
                self.P_pair_history.append(P_pair_out)
                self.b0_history.append(b0_out)
                self._trim_history()
            except Exception as e_hist:
                neuron_logger.warning(f"N {self.name}: Error updating history: {e_hist}")
        
        # --- Step 5: Return ---
        # Return z_out, which was calculated ONLY if the pair was structurally updated
        # If modulation_signal was 0, then z_out will be -1 (its initial value)
        # If the pair was ignored (v1=-1), z_out will also be -1.
        return S_out, P_pair_out, F_out, b0_out, z_out, P1_out, P2_out
    
    def plot_metrics_history(self, save_path: Optional[Path] = None):
        """
        Plots and optionally saves history graphs of the neuron's main metrics.

        Visualises time series of S, S_expected, FS (Familiarity Score),
        P_pair, F (Free Energy, log scale), and b0 (Number of components).
        Uses data from the `*_history` attributes.

        Args:
            save_path: Path (Path object) for saving the graph file. If None,
                       the graph is displayed on the screen.
        """
        
        history_lengths = [len(getattr(self, h, [])) for h in
                           ['S_history', 'F_history', 'P_pair_history', 'b0_history']]
        if not all(history_lengths):  # If at least one history is empty
            print(f"Neuron {self.name}: One or more history lists are empty. Cannot plot.")
            return
        min_len = min(l for l in history_lengths if l > 0)  # Minimum NON-EMPTY length
        if min_len < 2:
            print(f"Neuron {self.name}: Not enough history data (min_len={min_len}). Cannot plot.")
            return
        
        steps = list(range(min_len))
        # Get the last min_len elements
        S_hist = self.S_history[-min_len:]
        F_hist = self.F_history[-min_len:]
        P_hist = self.P_pair_history[-min_len:]
        b0_hist = self.b0_history[-min_len:]
        
        # Calculation of FS and S_expected
        FS_hist = [calculate_familiarity_score(f, k=self.hyperparameters.k_familiarity) for f in F_hist]
        S_exp_hist = [np.mean(S_hist[:max(1, i + 1)]) for i in range(min_len)]
        
        # Plotting graphs
        try:
            fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
            fig.suptitle(f"Neuron {self.name} Metrics History ({min_len} steps)", fontsize=16)
            # ... (plotting code as before, using axs[0]..axs[3]) ...
            # Graph 1: S and S_expected
            axs[0].plot(steps, S_hist, label='S', marker='.', ms=3)
            axs[0].plot(steps, S_exp_hist, label='S Expected', ls='--')
            axs[0].set_ylabel('Stability')
            axs[0].legend()
            axs[0].grid(True)
            axs[0].set_ylim(-0.1, 1.1)
            # Graph 2: FS and P_pair
            ax1_twin = axs[1].twinx()
            l1, = axs[1].plot(steps, FS_hist, label='FS', color='g', marker='.', ms=3)
            l2, = ax1_twin.plot(steps, P_hist, label='P_pair', color='b', marker='.', ls=':', ms=3, alpha=0.7)
            axs[1].set_ylabel('FS', color='g')
            ax1_twin.set_ylabel('P_pair', color='b')
            axs[1].set_ylim(-0.1, 1.1)
            ax1_twin.set_ylim(-0.1, 1.1)
            axs[1].legend(handles=[l1, l2], loc='upper left')
            axs[1].grid(True)
            # Graph 3: F
            axs[2].plot(steps, F_hist, label='F', color='r', marker='.', ms=3)
            axs[2].set_ylabel('Free Energy (F)')
            axs[2].legend()
            axs[2].grid(True)
            axs[2].set_yscale('symlog', linthresh=0.1)
            axs[2].set_ylim(bottom=-0.01)
            # Graph 4: b0
            axs[3].plot(steps, b0_hist, label='b0', drawstyle='steps-post')
            axs[3].set_ylabel('b0')
            axs[3].legend()
            axs[3].grid(True)
            axs[3].set_xlabel('Time Step')
            max_b0 = max(b0_hist) if b0_hist else 1
            if max_b0 < 20:
                axs[3].set_yticks(np.arange(0, max_b0 + 2, 1))
            else:
                axs[3].set_yticks(np.linspace(0, max_b0 + 1, 10, dtype=int))  # Otherwise, set fewer ticks
            
            plt.tight_layout(rect=(0., 0.03, 1., 0.97))
            if save_path:
                plt.savefig(save_path)
                print(f"N {self.name}: Plot saved to {save_path}")
                plt.close(fig)
            else:
                plt.show()
        except Exception as e_plot:
            print(f"Error N {self.name}: Failed plotting metrics: {e_plot}")
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                plt.close(fig)  # Close figure on error
    
    def dump(self):
        """
        Saves the current state of the neuron (complex, frequencies, maps, histories,
        parameters) to a binary file using `self.memory` (binstorage).
        The filename is determined as `{self.storage_path}/{self.name}.bin`.
        """
        
        # initial_memory = {
        #     'complex': SimplexTree(),
        #     'freq': defaultdict(int),
        #     'max_freq': 0,
        #     'max_history_len': 1000,
        #     'S_history': [],
        #     'F_history': [],
        #     'P_pair_history': [],
        #     'b0_history': [],
        #     'edge_weights': defaultdict(float)
        # }
        
        # Update data in the dictionary before saving
        self.memory.data['complex'] = self.complex
        self.memory.data['freq'] = self.freq  # defaultdict will be saved as dict
        self.memory.data['max_freq'] = self.max_freq
        self.memory.data['max_history_len'] = self.max_history_len
        self.memory.data['S_history'] = self.S_history
        self.memory.data['F_history'] = self.F_history
        self.memory.data['P_pair_history'] = self.P_pair_history
        self.memory.data['b0_history'] = self.b0_history
        self.memory.data['edge_weights'] = self.edge_weights
        
        try:
            self.memory.dump()
        except Exception as e:
            print(f"Error N {self.name}: Failed dumping data: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    from dictionary_methods import read_dictionary
    
    prepared_dictionary_name = 'ru_dict.txt'
    input_path = '../assets/lang/ru'
    
    got_bigrams = read_dictionary(prepared_dictionary_name, input_path)
    # Example of use
    neuron = SimplicialNeuron(0, Path('./data/neurons'), hyperparameters=NeuronHyperParams)
    
    for step, (a, b) in enumerate(got_bigrams):
        # if step >= 100:
        #     break
        S, P, F, b0, z, el1, el2 = neuron.ascending_activation(ord(a), ord(b))
        ps = P / (S + NeuronHyperParams.epsilon)
        print(
            f"Learning Pair ({a}, {b}): S={S:.3f}, P={P:.3f}, F={F:.3f}, b0={b0:.3f}, P/S={ps:.3f}, z={z}, P_el1={el1}, P_el2={el2}")
    #
    # for (a, b) in got_bigrams:
    #     S, P, F, b0, z, el1, el2 = neuron.ascending_activation(a, b)
    #     ps = P / (S + EPSILON)
    #     print(f"Test Pair ({a}, {b}): S={S:.3f}, P={P:.3f}, F={F:.3f}, b0={b0:.3f}, P/S={ps:.3f}, z={z}, P_el1={el1}, P_el2={el2}")
    
    a, b = ('D', 'e')
    S, P, F, b0, z, el1, el2 = neuron.ascending_activation(ord(a), ord(b))
    ps = P / (S + NeuronHyperParams.epsilon)
    print(
        f"Learning Pair ({a}, {b}): S={S:.3f}, P={P:.3f}, F={F:.3f}, b0={b0:.3f}, P/S={ps:.3f}, z={z}, P_el1={el1}, P_el2={el2}")
    # for _ in range(0, 100):
    #     S, P, F, b0, z = neuron.ascending_activation(a, b)
    #     epsilon = 1e-9
    #     ps = P / (S + epsilon)
    #     print(f"Learning Pair ({a}, {b}): S={S:.3f}, P={P:.3f}, F={F:.3f}, b0={b0:.3f}, P/S={ps:.3f}, z={z}")
    
    storage_dir_n = Path('./results/early/neuron_plots_data')
    plot_file_path = storage_dir_n / f'neuron_{0}_metrics.png'
    neuron.plot_metrics_history(save_path=plot_file_path)
    
    # neuron.visualize_P_S_relation()
    # neuron.visualize_topology()
    neuron.dump()