# SPDX-License-Identifier: LicenseRef-NonCommercial-Research
# Copyright (c) 2025 Alexei Firssoff. ORCID: 0009-0006-0316-116X

"""
HELOS: Hierarchical Emergence of Latent Ontological Structure
Author: Alexei Firssoff
License: See LICENSE.txt for terms and conditions
"""


import math
import heapq
from typing import (
    List,
    Tuple,
    Dict,
    Union,
    Optional,
    Set,
    Any
)
from dataclasses import (
    dataclass,
    field
)
import time
from collections import defaultdict
from graphviz import Digraph
import os


# --- Predictive Model (Placeholder with Frequency Logic) ---
@dataclass
class PredictiveModel:
    """
    A placeholder predictive model based on frequency counts with Laplace smoothing.
    In a full FEP implementation, this would be a more complex probabilistic model.
    It estimates the 'surprise' (-log probability) of observing a predicted node
    given a context.
    """
    counts: Dict[Tuple[Optional[int], ...], float] = field(default_factory=lambda: defaultdict(float))
    context_totals: Dict[Tuple[Optional[int], ...], float] = field(default_factory=lambda: defaultdict(float))
    total_observations: float = 0.0
    smoothing: float = 0.01
    vocabulary_size: int = 100  # Estimate, dynamically updated
    
    def get_surprise(self, predicted_node: 'Node', *context_nodes: 'Node') -> float:
        """
        Calculates the surprise (-log P(predicted_node | context_nodes)).
        Lower values indicate a better prediction (less surprise).

        Args:
            predicted_node (Node): The node whose appearance we are evaluating.
            *context_nodes (Node): The preceding nodes forming the context.

        Returns:
            float: The calculated surprise value (-log P).
        """
        # Use hashes for dictionary keys
        context_key = tuple(hash(n) for n in context_nodes if isinstance(n, Node))
        predicted_key = hash(predicted_node) if isinstance(predicted_node, Node) else hash(str(predicted_node))
        
        full_key = context_key + (predicted_key,)
        
        count = self.counts.get(full_key, 0.0)
        context_total = self.context_totals.get(context_key, 0.0)
        
        # Dynamically estimate vocabulary size (very rough)
        vocab_size = max(10, len(node_registry) // 10 + len(self.counts) // 5 + 10)
        
        # Laplace Smoothed Probability P = (count + k) / (N + k*V)
        prob = (count + self.smoothing) / (context_total + self.smoothing * vocab_size)
        
        # Surprise = -log P
        surprise = -math.log(prob + 1e-9)  # Add epsilon for numerical stability
        
        # Bound the maximum surprise to prevent infinite values for unseen events
        max_surprise = -math.log(self.smoothing / (context_total + self.smoothing * vocab_size) + 1e-9)
        
        # Return capped surprise (add minor noise?)
        return max(0.01, min(surprise, max_surprise * 1.2))  # Add random.uniform(-0.01, 0.01)?
    
    def update(self, update_strength: float, predicted_node: 'Node', *context_nodes: 'Node'):
        """
        Updates the frequency counts of the model based on an observation.
        The update_strength reflects the confidence or relevance of this observation
        (e.g., derived from the success signal in Bayesian update).

        Args:
            update_strength (float): Factor indicating the strength of this update (e.g., ~1.0 for success).
            predicted_node (Node): The node that was observed.
            *context_nodes (Node): The context in which the node was observed.
        """
        context_key = tuple(hash(n) for n in context_nodes if isinstance(n, Node))
        predicted_key = hash(predicted_node) if isinstance(predicted_node, Node) else hash(str(predicted_node))
        full_key = context_key + (predicted_key,)
        
        # Increment counts proportionally to the update strength
        self.counts[full_key] += update_strength
        self.context_totals[context_key] += update_strength
        self.total_observations += update_strength
        # Update vocab size estimate
        self.vocabulary_size = max(self.vocabulary_size, len(self.counts) + 5)


# --- Core Recursive Node Structure ---
NodeContent = Union[str, Tuple['Node', 'Node']]


@dataclass(frozen=True)  # Nodes are immutable, identified by content
class Node:
    """
    Represents a node in the hierarchical binary parse tree.
    Corresponds to the universal recursive type S.
    It can be a leaf (string) or an internal node (pair of Nodes).
    """
    content: NodeContent
    _hash: Optional[int] = field(default=None, repr=False, compare=False)
    
    def __hash__(self):
        """Precompute hash for efficiency, as Node is immutable."""
        if self._hash is None:
            # Use standard tuple/string hash on content
            object.__setattr__(self, '_hash', hash(self.content))
        return self._hash
    
    def __repr__(self):
        """Provides a compact parenthesised representation, avoiding excessive depth."""
        MAX_REPR_DEPTH = 8
        memo = {}  # Track visited nodes during representation building for this specific call
        node_map = {}  # Map node hashes to simple IDs for concise output
        counter = 0
        
        def get_node_id(node):
            nonlocal counter
            h = hash(node)
            if h not in node_map:
                node_map[h] = counter
                counter += 1
            return node_map[h]
        
        def _repr_recursive(node, depth):
            node_hash = hash(node)
            # Use id() for memo during this specific repr call to handle distinct objects with the same content if needed
            node_id_local = id(node)
            if node_id_local in memo:
                return f"{node}"  # Show node ID if repeated in this structure
            if depth > MAX_REPR_DEPTH:
                return "..."
            memo[node_id_local] = True
            
            res = "..."
            if isinstance(node.content, str):
                res = node.content  # Leaf node
            elif isinstance(node.content, tuple):
                # Internal node: recursively represent children
                l = node.content[0]
                r = node.content[1]
                left_repr = _repr_recursive(l, depth + 1) if isinstance(l, Node) else repr(l)
                right_repr = _repr_recursive(r, depth + 1) if isinstance(r, Node) else repr(r)
                res = f"({left_repr}, {right_repr})"
            else:
                res = repr(node.content)  # Fallback
            
            # Don't delete from the memo here to show cycles/repeats with IDs
            return res
        
        return _repr_recursive(self, 0)
    
    def get_leaves(self) -> List[str]:
        """ Extracts the sequence of leaf nodes (graphemes) from the tree. """
        leaves = []
        queue = [self]
        visited_hashes = set()
        count = 0
        MAX_LEAVES_NODES = 2000
        while queue and count < MAX_LEAVES_NODES:
            node = queue.pop(0)
            node_hash = hash(node)
            if node_hash in visited_hashes:
                continue
                
            visited_hashes.add(node_hash); count += 1
            if isinstance(node.content, str):
                leaves.append(node.content)
            elif isinstance(node.content, tuple):
                # Add children to the queue for BFS traversal
                if isinstance(node.content[0], Node):
                    queue.append(node.content[0])
                if isinstance(node.content[1], Node):
                    queue.append(node.content[1])
        # if count >= MAX_LEAVES_NODES:
        #      print("Warning: Max node limit reached in get_leaves.")
        return leaves


# --- Global Node Registry (Memory M) ---
# Maps node hash to its parameters (FE, predictors, stats)
node_registry: Dict[int, Dict[str, Any]] = {}
node_creation_counter = 0  # For penalizing new nodes

# --- Constants and Parameters ---
INITIAL_LOG_PRIOR = -math.log(1000.0)  # Initial -log p(Node), low probability
COMPLEXITY_PENALTY_PAIR = 0.1  # Cost added to Surprise for a composition operation
LEARNING_RATE_PRIOR = 0.07  # Learning rate for updating log_prior
PRIOR_UPDATE_FACTOR = 0.7  # Factor scaling the success signal for predictor updates


# --- Registry Management Functions ---
def get_or_create_node_data(content: NodeContent) -> Tuple[Node, Dict[str, Any]]:
    """
    Retrieves or creates a node and its associated data in the registry.
    Handles initialisation of parameters for new nodes.

    Args:
        content: The content of the node (string or tuple of Nodes).

    Returns:
        A tuple containing the Node object and its data dictionary from the registry.
    """
    global node_creation_counter
    node = Node(content)
    node_hash = hash(node)
    if node_hash not in node_registry:
        node_creation_counter += 1
        # Initialise parameters for a new node
        node_registry[node_hash] = {
            'node_obj': node,  # Store the node object itself
            # Initial prior reflects novelty penalty
            'log_prior': INITIAL_LOG_PRIOR - math.log(node_creation_counter + 1),
            'freq': 0,
            'predictor_next': PredictiveModel(),  # Placeholder model
            'predictor_internal': PredictiveModel(),  # Placeholder model
        }
    return node_registry[node_hash]['node_obj'], node_registry[node_hash]


def get_node_log_prior(node: Node) -> float:
    """ Safely retrieves the log prior probability of a node from the registry. """
    node_data = node_registry.get(hash(node))
    # Return a very low log prior if a node is unknown (high complexity)
    return node_data['log_prior'] if node_data else INITIAL_LOG_PRIOR - 10.0


def update_log_prior(node: Node, surprise_signal: float):
    """
    Updates the log_prior of a node based on the surprise signal from inference.
    Lower surprise (better prediction) should increase the log_prior (closer to 0).
    This approximates Bayesian updating of the node's prior probability.

    Args:
        node (Node): The node to update.
        surprise_signal (float): The average surprise associated with parses involving this node.
                                Lower values indicate the node participated in better explanations.
    """
    node_hash = hash(node)
    node_data = node_registry.get(node_hash)
    if not node_data:
        return
    
    # Convert surprise signal (lower is better) to success signal (higher is better)
    # Use exponential decay based on surprise relative to initial surprise level
    normalized_surprise = max(0, surprise_signal / abs(INITIAL_LOG_PRIOR + 1e-6))
    success_signal = math.exp(-normalized_surprise)  # Ranges from 1 (low surprise) down to 0
    
    # Target log_prior: moves from INITIAL_LOG_PRIOR towards 0 based on success
    target_log_prior = INITIAL_LOG_PRIOR * (1 - success_signal)
    
    # Update log_prior towards target
    delta = target_log_prior - node_data['log_prior']
    node_data['log_prior'] += LEARNING_RATE_PRIOR * delta
    
    # Clamp log_prior to be slightly negative (probabilities <= 1)
    node_data['log_prior'] = min(-0.01, node_data['log_prior'])
    
    node_data['freq'] += 1  # Increment frequency count
    
    # Update predictive models (using success signal as strength)
    # Note: This needs the actual context of the prediction for the predictor update.
    # The current implementation of PredictiveModel.update is simplified.
    update_strength = success_signal * PRIOR_UPDATE_FACTOR
    current_node_obj = node_data['node_obj']
    if node_data['predictor_next']:
        # Requires context and predicted node for proper update
        pass  # node_data['predictor_next'].update(update_strength, context..., predicted...)
    if node_data['predictor_internal']:
        if isinstance(current_node_obj.content, tuple):
            left_node = current_node_obj.content[0] if isinstance(current_node_obj.content[0], Node) else None
            right_node = current_node_obj.content[1] if isinstance(current_node_obj.content[1], Node) else None
            if left_node and right_node:
                # Requires context for update
                pass  # node_data['predictor_internal'].update(update_strength, left_node, right_node)


# --- Tree Complexity Calculation ---
def get_tree_complexity(node: Node) -> float:
    """
    Calculates the total complexity of a parse tree.
    Complexity(T) = Sum over nodes N_k in T of Complexity(N_k)
    Complexity(N_k) is approximated by -log P_prior(N_k).
    """
    complexity = 0.0
    queue = [node]
    visited_hashes = set()
    count = 0
    MAX_TRAVERSE_NODES = 5000
    while queue and count < MAX_TRAVERSE_NODES:
        current_node = queue.pop(0)
        node_hash = hash(current_node)
        if node_hash in visited_hashes:
            continue
            
        visited_hashes.add(node_hash); count += 1
        
        # Add complexity of the current node (-log prior)
        complexity += -get_node_log_prior(current_node)
        
        if isinstance(current_node.content, tuple):
            # Add children to the queue
            if isinstance(current_node.content[0], Node):
                queue.append(current_node.content[0])
            if isinstance(current_node.content[1], Node):
                queue.append(current_node.content[1])
    # Optional: Add penalty based on tree size (number of nodes)
    # complexity += len(visited_hashes) * COMPLEXITY_PENALTY_NODE
    return complexity


# --- Search State for Beam Search ---
@dataclass(order=False)
class SearchState:
    """ Represents a hypothesis during the parse search. Ordered by FE for min-heap. """
    fe: float = field(compare=True)  # Total Free Energy (F = Complexity + Surprise)
    priority: float = field(init=False, compare=False)  # Alias for fe for heap
    complexity: float = field(compare=False)  # Complexity component (-Sum log P_prior)
    node: Node = field(compare=False)  # Root node of the parsed subtree
    start_idx: int = field(compare=False)  # Start index in the original sequence
    end_idx: int = field(compare=False)  # End index in the original sequence
    surprise: float = field(default=0.0, compare=False)  # Surprise component (-Sum log P_likelihood)
    
    def __post_init__(self):
        self.priority = self.fe  # Lower FE is of higher priority
    
    def __lt__(self, other):
        """ Comparison for heapq (min-heap based on FE). """
        if not isinstance(other, SearchState):
            return NotImplemented
        # Primary sort by FE, secondary by complexity (prefer simpler if FE is equal)
        if self.fe != other.fe:
            return self.fe < other.fe
        return self.complexity < other.complexity


# --- FEP-Guided Parser (CKY-like Beam Search) ---
def find_best_parses_fep(sequence: List[str], beam_width: int = 5) -> List[SearchState]:
    """
    Finds the best parse trees for a sequence using a CKY-like dynamic programming
    approach with beam search, guided by an approximate Free Energy calculation.

    Args:
        sequence (List[str]): The input sequence of graphemes.
        beam_width (int): The maximum number of hypotheses to keep at each step.

    Returns:
        List[SearchState]: A list of the best hypotheses covering the entire sequence,
                           sorted by Free Energy (the best first).
    """
    n = len(sequence)
    # Chart stores best hypotheses for spans (i, j)
    chart: Dict[Tuple[int, int], List[SearchState]] = {}
    
    # Initialise diagonal (single characters)
    for i in range(n):
        leaf_node, leaf_data = get_or_create_node_data(sequence[i])
        leaf_complexity = -leaf_data['log_prior']  # Complexity = -log P_prior
        leaf_surprise = 0.0  # No prediction error for a leaf itself
        initial_fe = leaf_complexity + leaf_surprise
        hypo = SearchState(fe=initial_fe, complexity=leaf_complexity, surprise=leaf_surprise,
                           node=leaf_node, start_idx=i, end_idx=i)
        chart[(i, i)] = [hypo]
    
    # Fill the chart for spans of increasing length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            beam: List[SearchState] = []  # Use min-heap for beam
            
            # Consider all split points k
            for k in range(i, j):
                left_hypotheses = chart.get((i, k), [])
                right_hypotheses = chart.get((k + 1, j), [])
                
                if not left_hypotheses or not right_hypotheses:
                    continue
                
                # Combine hypotheses from left and right spans
                for h_left in left_hypotheses[:beam_width]:  # Limit combinations
                    for h_right in right_hypotheses[:beam_width]:
                        if not isinstance(h_left.node, Node) or not isinstance(h_right.node, Node):
                            continue
                        
                        # 1. Create a new parent node
                        new_node, new_node_data = get_or_create_node_data((h_left.node, h_right.node))
                        
                        # 2. Calculate Surprise added at this step (Prediction Error)
                        left_node_data = node_registry.get(hash(h_left.node))
                        
                        # Surprise from left predicting right
                        pe_left = -INITIAL_LOG_PRIOR
                        if left_node_data and left_node_data.get('predictor_next'):
                            try:
                                pe_left = left_node_data['predictor_next'].get_surprise(h_right.node)
                            except Exception:
                                pass
                        
                        # Surprise from a new node predicting its children
                        pe_internal = -INITIAL_LOG_PRIOR
                        if new_node_data.get('predictor_internal'):
                            try:
                                pe_internal = new_node_data['predictor_internal'].get_surprise(h_left.node,
                                                                                               h_right.node)
                            except Exception:
                                pass
                        
                        # Surprise added at this composition step
                        current_step_surprise = (pe_left + pe_internal) / 2.0 + COMPLEXITY_PENALTY_PAIR
                        
                        # Total Surprise for the new hypothesis
                        total_surprise = h_left.surprise + h_right.surprise + current_step_surprise
                        
                        # 3. Calculate Complexity for the new hypothesis
                        # Complexity = Sum Complexity(children) + Complexity(new node)
                        complexity_new_node = -new_node_data['log_prior']
                        total_complexity = h_left.complexity + h_right.complexity + complexity_new_node
                        
                        # 4. Calculate Total Free Energy
                        total_fe = total_complexity + total_surprise
                        
                        # 5. Add to Beam Search heap
                        new_hypo = SearchState(fe=total_fe, complexity=total_complexity, surprise=total_surprise,
                                               node=new_node, start_idx=i, end_idx=j)
                        
                        if len(beam) < beam_width:
                            heapq.heappush(beam, new_hypo)
                        elif new_hypo.fe < beam[0].fe:  # If better than the worst in beam
                            heapq.heapreplace(beam, new_hypo)  # Replace worst
            
            # Store the best hypotheses for this span
            if beam:
                chart[(i, j)] = sorted(beam, key=lambda x: x.fe)  # Keep the chart sorted
    
    # Return best hypotheses for the full span [0, n-1]
    return chart.get((0, n - 1), [])


# --- Graphviz Visualization ---
output_dir = "../data/results/morph_parse_trees"  # New directory
os.makedirs(output_dir, exist_ok=True)


def visualize_tree(node: Node, filename: str):
    """ Generates a PNG visualization of the parse tree using Graphviz. """
    dot = Digraph(comment='Parse Tree', node_attr={'shape': 'record', 'fontsize': '10'})
    dot.attr(rankdir='TB', size="8,10")  # Adjust size if needed
    node_counter = 0
    node_map = {}
    
    def get_viz_id(n):
        # Creates short, unique IDs for graphviz nodes
        nonlocal node_counter
        h = hash(n)
        if h not in node_map:
            node_map[h] = f"N{node_counter}"
            node_counter += 1
        return node_map[h]
    
    def add_nodes_edges(current_node: Union[Node, str]):
        # Check if node object is valid before proceeding
        if not isinstance(current_node, Node):
            # Handle non-Node types if they somehow appear (e.g., basic strings in pairs)
            fallback_id = f"fallback_{get_viz_id(current_node)}"
            dot.node(fallback_id, label=f"'{repr(current_node)}'", shape='box', style='filled', color='red')
            return fallback_id
        
        viz_id = get_viz_id(current_node)
        
        # Avoid re-adding nodes and edges if already processed (for non-tree graphs, though ours are trees)
        # if viz_id in added_nodes: return viz_id
        # added_nodes.add(viz_id)
        
        node_data = node_registry.get(hash(current_node))
        label_parts = []
        
        if isinstance(current_node.content, str):
            label_parts.append(f"'{current_node.content}'")
            shape = 'plaintext'
        elif isinstance(current_node.content, tuple):
            label_parts.append("()")  # Pair node
            shape = 'circle'
        else:
            label_parts.append("?")
            shape = 'diamond'
        
        if node_data:
            label_parts.append(f"Fq:{node_data['freq']}")
            label_parts.append(f"LP:{node_data['log_prior']:.1f}")
        else:
            label_parts.append("Fq:?")
            label_parts.append("LP:?")
        
        # Create a node label using Graphviz record shape for multiple lines
        dot.node(viz_id, label="{ " + " | ".join(label_parts) + " }", shape=shape)
        
        if isinstance(current_node.content, tuple):
            # Recursively add children and edges
            left_child_viz_id = add_nodes_edges(current_node.content[0])
            right_child_viz_id = add_nodes_edges(current_node.content[1])
            dot.edge(viz_id, left_child_viz_id)
            dot.edge(viz_id, right_child_viz_id)
        
        return viz_id
    
    # added_nodes = set() # Track added nodes to prevent duplicates in graphviz output if needed
    if isinstance(node, Node):
        add_nodes_edges(node)
    else:
        print(f"Error: visualize_tree called with non-Node object: {node}")
        return
    
    # Save and render
    try:
        filepath = os.path.join(output_dir, filename)
        # Use engine='dot' explicitly if needed
        dot.render(filepath, format='png', view=False, cleanup=True, engine='dot')
        print(f"  Tree visualization saved to {filepath}.png")
    except Exception as e:
        print(f"  Graphviz Error: {e}. Check Graphviz installation and PATH.")
        # print(f"  DOT source:\n{dot.source}")


# --- Main Processing Function ---
def process_word_fep_v3(word: str, update_weights: bool = True, top_k: int = 3, beam_width: int = 10,
                        visualize: bool = False) -> List[SearchState]:
    """ Analyzes a word using FEP-guided parsing and optionally updates weights. """
    letters = list(word)
    if not letters:
        return []
    
    print(f"\nAnalyzing (FEP V3 Final): {word}")
    start_time = time.time()
    # Ensure beam_width for search is sufficient
    search_beam_width = max(top_k, beam_width, 5)
    best_hypotheses = find_best_parses_fep(letters, beam_width=search_beam_width)
    end_time = time.time()
    print(f"  Parse search took: {end_time - start_time:.3f} sec, found {len(best_hypotheses)} hypotheses.")
    
    if not best_hypotheses:
        print(f"  No parses found for '{word}'.")
        return []
    
    print(f"  Top {min(top_k, len(best_hypotheses))} hypotheses for '{word}':")
    for i, hypo in enumerate(best_hypotheses[:top_k]):
        # Recalculate complexity for printing consistency
        complexity = get_tree_complexity(hypo.node)
        surprise = hypo.surprise
        recalc_fe = complexity + surprise
        node_repr = repr(hypo.node)
        if len(node_repr) > 150:
            node_repr = node_repr[:75] + "..." + node_repr[-75:]
        print(
            f"    {i + 1}. Parse: {node_repr} (FE: {hypo.fe:.2f} ~ Recalc FE: {recalc_fe:.2f} = Comp: {complexity:.2f} + Surp: {surprise:.2f})")
        
        # Visualize only the best hypothesis
        if i == 0 and visualize:
            filename = f"{word}_best_parse"
            # Pass the actual Node object to visualize
            if isinstance(hypo.node, Node):
                visualize_tree(hypo.node, filename)
            else:
                print(f"Error: Cannot visualize non-Node object: {hypo.node}")
    
    # Update weights based on the best hypothesis
    if update_weights and best_hypotheses:
        best_hypo = best_hypotheses[0]
        
        # Gather all unique nodes from the best parse tree
        nodes_to_update: Set[Node] = set()
        queue = [best_hypo.node]
        visited_hashes = set()
        count = 0
        MAX_UPDATE_NODES = 5000
        while queue and count < MAX_UPDATE_NODES:
            current_node = queue.pop(0)
            node_hash = hash(current_node)
            if node_hash in visited_hashes:
                continue
                
            visited_hashes.add(node_hash)
            count += 1
            node_data = node_registry.get(node_hash)
            if node_data:
                nodes_to_update.add(node_data['node_obj'])  # Add the Node object
                if isinstance(current_node.content, tuple):
                    if isinstance(current_node.content[0], Node):
                        queue.append(current_node.content[0])
                    if isinstance(current_node.content[1], Node):
                        queue.append(current_node.content[1])
        
        # Use average surprise as the error signal for updating priors
        average_surprise = best_hypo.surprise / (len(nodes_to_update) if nodes_to_update else 1)
        # print(f"  Updating {len(nodes_to_update)} nodes with avg surprise signal: {average_surprise:.2f}")
        
        for node_obj in nodes_to_update:
            # Update log_prior and potentially predictor stats (implicitly via PredictiveModel.update)
            update_log_prior(node_obj, average_surprise)
            
            # Return the top_k hypotheses found
    return best_hypotheses[:top_k]