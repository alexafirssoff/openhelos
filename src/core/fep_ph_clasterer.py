# SPDX-License-Identifier: LicenseRef-NonCommercial-Research
# Copyright (c) 2025 Alexei Firssoff. ORCID: 0009-0006-0316-116X

"""
LifelongClusterer: Topology-Guided Symbolic Clustering Based on Co-occurrence and Persistent Homology

This module implements a lifelong clustering algorithm for symbolic sequences,
designed to autonomously group frequently co-occurring symbols into higher-level
representations (clusters). It draws on principles from topological data analysis,
specifically persistent homology, to determine the structural stability of potential
clusters in an evolving symbolic graph.

Key Features:
- Maintains a co-occurrence graph of symbols and previously formed clusters.
- Detects candidate clusters based on connectivity and symbol frequency.
- Computes persistence diagrams for each candidate cluster using Gudhi's simplex trees.
- Evaluates cluster stability via temporal persistence diagram similarity.
- Symbolises stable clusters, replacing constituent symbols with unique cluster identifiers.
- Supports dynamic, online processing of symbolic input streams (lifelong learning).
- Offers introspection via `get_state()` for external analysis or visualisation.

Theoretical Context:
While this implementation does not explicitly compute variational free energy,
its design reflects core ideas of the Free Energy Principle (FEP) in an emergent and heuristic form:
- Symbolisation reduces structural uncertainty and can be viewed as entropy minimisation.
- The formation of stable, compact representations implicitly performs model compression.
- The evolving symbolic hierarchy may be interpreted as a generative model adapting to statistical regularities.

This system is part of the broader OpenHELOS framework and is intended to empirically explore theoretical predictions
regarding autonomous structure formation, abstraction, and symbolic emergence from local statistical interactions.

Author: Alexei Firssoff
Year: 2025
License: Non-Commercial Research (see LICENSE.txt for details)
"""

from collections import (
    defaultdict,
    Counter
)
from typing import (
    Any,
    Set,
    Dict,
    List,
    Tuple,
    FrozenSet,
    Optional
)

import gudhi as gd
from gudhi.hera import (
    wasserstein_distance,
    bottleneck_distance
)
import numpy as np
import logging
import json  # For get_state output

# Set logging to DEBUG to see stability check details
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper type hints
Node = Any  # Can be a symbol (str) or a cluster ID (int)
PersistenceDiagram = List[Tuple[float, float]]  # List of birth-death pairs


class LifelongClusterer:
    """
    Performs lifelong emergent clustering of symbols based on pairwise
    co-occurrence and topological stability (Persistent Homology).
    Clusters can be symbolized and participate in further co-occurrence tracking.

    Uses a heuristic to track the stability of the largest symbol-based component.
    """
    
    def __init__(self,
                 # RELAXED parameters for testing the heuristic
                 stability_threshold: float = 0.5,  # TRY HIGHER VALUE (e.g., 0.5 or 1.0)
                 persistence_history_length: int = 3,  # How many diagrams to compare
                 required_stability_passes: int = 2,  # TRY 1
                 min_cluster_size: int = 5,  # Increased slightly for alphabet
                 min_cooccurrence_for_ph: int = 2,  # TRY 1
                 max_filtration_value: float = 200.0,  # For converting counts to filtration
                 symbolize_immediately: bool = False  # For debugging: symbolize after first PH calc
                 ):
        """
        Initializes the clusterer.

        Args:
            stability_threshold: Max persistence diagram distance to be considered stable.
            persistence_history_length: Number of recent diagrams to check for stability.
            required_stability_passes: Number of consecutive checks needed for symbolization.
            min_cluster_size: Minimum number of original symbols required to form a cluster.
            min_cooccurrence_for_ph: Minimum co-occurrence count to consider an edge for PH.
            max_filtration_value: Used in count -> filtration value conversion.
            symbolize_immediately: If True, symbolizes any potential cluster right away.
        """
        self.stability_threshold = stability_threshold
        self.persistence_history_length = persistence_history_length
        self.required_stability_passes = required_stability_passes  # Store this new param
        self.min_cluster_size = min_cluster_size
        self.min_cooccurrence_for_ph = min_cooccurrence_for_ph
        self.max_filtration_value = max_filtration_value
        self.symbolize_immediately = symbolize_immediately
        
        # --- Core State ---
        self.cooccurrence_counts: Dict[Node, Counter[Node]] = defaultdict(Counter)
        self.node_total_occurrences: Counter[Node] = Counter()
        
        self.symbols: Set[str] = set()
        self.cluster_ids: Set[int] = set()
        self.next_cluster_id: int = 0
        
        self.cluster_definitions: Dict[int, FrozenSet[str]] = {}
        self.symbol_to_cluster: Dict[str, int] = {}
        
        # --- NEW: Tracking potential clusters (using fixed keys for heuristics) ---
        # Key: string identifier (e.g., "main_candidate")
        # Value: Dict with 'persistence_history': List[Dict[dim, PersistenceDiagram]],
        #                 'stability_checks_passed': int,
        #                 'current_members': FrozenSet[Node]
        self.potential_clusters: Dict[str, Dict] = {}  # Changed from FrozenSet key
        
        logging.info(
            f"LifelongClusterer initialized with threshold={stability_threshold}, history={persistence_history_length}, passes={required_stability_passes}, min_ph_cooc={min_cooccurrence_for_ph}")
    
    def _get_node_representation(self, symbol: str) -> Node:
        """Returns the cluster ID if the symbol is clustered, otherwise the symbol itself."""
        return self.symbol_to_cluster.get(symbol, symbol)
    
    def process_string(self, text: str):
        """
        Processes a string, updating co-occurrence counts between nodes
        (symbols or cluster IDs).
        """
        if not text:
            return
        
        if text.endswith('\n'):
            text = text[:-1]
        
        current_symbols = set(text)
        new_symbols = current_symbols - self.symbols
        for sym in new_symbols:
            if sym not in self.symbol_to_cluster:
                self.symbols.add(sym)
        
        nodes_involved_in_update = set()  # Track nodes (syms or IDs) affected
        for i in range(len(text) - 1):
            sym1 = text[i]
            sym2 = text[i + 1]
            
            if sym1 == sym2:
                continue
            
            node1 = self._get_node_representation(sym1)
            node2 = self._get_node_representation(sym2)
            
            self.cooccurrence_counts[node1][node2] += 1
            self.cooccurrence_counts[node2][node1] += 1
            # Only count occurrence once per pair processing? Or once per symbol appearance?
            # Let's stick to once per involvement in a pair.
            if i == 0:  # Count first symbol once
                self.node_total_occurrences[node1] += 1
            self.node_total_occurrences[node2] += 1  # Count second symbol
            
            nodes_involved_in_update.add(node1)
            nodes_involved_in_update.add(node2)
        
        # Check for cluster updates focusing on affected nodes
        self._check_for_new_clusters(nodes_involved_in_update)
    
    def _build_simplex_tree(self, nodes_in_cluster: FrozenSet[Node]) -> Optional[gd.simplex_tree.SimplexTree]:
        """
        Builds a Gudhi SimplexTree for a set of nodes based on co-occurrence counts.
        Filtration value is inversely related to count (stronger edges appear earlier).
        Uses a stable filtration value calculation (independent of max_count).
        """
        st = gd.simplex_tree.SimplexTree()
        node_list = sorted(list(nodes_in_cluster), key=lambda x: str(x))
        node_to_vertex = {node: i for i, node in enumerate(node_list)}
        
        vertices_added = set()  # Keep track of added vertices
        
        # --- Pass 1: Add vertices and collect edges ---
        edges_to_add = []
        min_positive_filtration = self.max_filtration_value  # Track min value > 0 for later vertex insertion
        has_edges = False
        
        for i, node1 in enumerate(node_list):
            vertex1_idx = node_to_vertex[node1]
            # Add vertex implicitly later if involved in an edge,
            # or explicitly now if potentially isolated
            
            for j in range(i + 1, len(node_list)):
                node2 = node_list[j]
                vertex2_idx = node_to_vertex[node2]
                
                count1 = self.cooccurrence_counts[node1].get(node2, 0)
                count2 = self.cooccurrence_counts[node2].get(node1, 0)
                count = max(count1, count2)
                
                if count >= self.min_cooccurrence_for_ph:
                    has_edges = True
                    # --- NEW STABLE FILTRATION ---
                    # Option 1: Inverse count (handle count=0 implicitly via min_cooccurrence_for_ph)
                    # filtration_value = 1.0 / float(count)
                    # Option 2: Negative count (if Gudhi handles it, low val = early)
                    # filtration_value = -float(count)
                    # Option 3: Max Value - Count (Robust, ensures positive filtration)
                    filtration_value = self.max_filtration_value - float(count)
                    
                    # Ensure filtration value is positive if needed, but Max-Count should be fine
                    # if max_filtration_value is large enough. Add safety offset if needed.
                    filtration_value = max(0.001, filtration_value)  # Ensure slightly > 0
                    
                    edges_to_add.append(((vertex1_idx, vertex2_idx), filtration_value))
                    min_positive_filtration = min(min_positive_filtration, filtration_value)
                    vertices_added.add(vertex1_idx)
                    vertices_added.add(vertex2_idx)
        
        # --- Pass 2: Add vertices ---
        # Add vertices slightly before the first edge appears
        vertex_filtration = -0.01
        for i, node in enumerate(node_list):
            # Add all vertices consistently
            st.insert([i], filtration=vertex_filtration)
        
        # --- Pass 3: Add edges ---
        if not has_edges and len(node_list) < 2:
            logging.debug(f"Not enough edges/nodes ({len(node_list)} nodes) for PH in {nodes_in_cluster}")
            return None
        
        for edge_indices, filtration in edges_to_add:
            st.insert(list(edge_indices), filtration=filtration)
        
        # Gudhi recommends this step
        try:
            st.make_filtration_non_decreasing()
        except Exception as e:
            logging.warning(f"Could not make filtration non-decreasing for {nodes_in_cluster}: {e}")
            # Might indicate issues with filtration values (e.g., vertex filter > edge filter)
        
        return st
    
    def _calculate_persistence(self, st: gd.simplex_tree.SimplexTree) -> Dict[int, List[Tuple[float, float]]]:
        """
        Calculates persistence diagrams (birth-death interval pairs)
        for dimensions 0 and 1.
        """
        persistence_diagrams = {}
        try:
            st.compute_persistence()
            # Get persistence intervals for dimension 0 (connected components)
            persistence_diagrams[0] = st.persistence_intervals_in_dimension(0)
            # logging.debug(f"Dim 0 Intervals: {persistence_diagrams[0]}")
            
            # Get persistence intervals for dimension 1 (cycles/holes)
            if st.dimension() >= 1:
                persistence_diagrams[1] = st.persistence_intervals_in_dimension(1)
            else:
                persistence_diagrams[1] = []
        except Exception as e:
            logging.error(f"Error computing persistence: {e}")
            persistence_diagrams[0] = []
            persistence_diagrams[1] = []  # Return empty on error
        
        return persistence_diagrams
    
    def _get_diagram_points(self, intervals: List[Tuple[float, float]]) -> np.ndarray:
        """
        Extracts finite persistence points (birth, death) from Gudhi interval pairs.
        Input `intervals` is a list of (birth, death) tuples.
        """
        points = []
        if intervals is None:
            return np.empty((0, 2))
        for birth, death in intervals:
            # Filter out infinite persistence bars for stability comparison,
            # except for the main component in dim 0?
            # Let's filter all infinities for now for Wasserstein distance.
            if death != float('inf') and birth != death:  # Also filter zero-persistence points
                points.append([birth, death])
        return np.array(points) if points else np.empty((0, 2))
    
    # NEW helper method for stability check based on history list
    # В методе _check_stability_on_history
    def _check_stability_on_history(self, history: List[Dict[int, PersistenceDiagram]]) -> bool:
        if len(history) < self.persistence_history_length:
            logging.warning("Stability check called with insufficient history.")
            return False
        
        recent_diagrams = history[-self.persistence_history_length:]
        overall_stable = True
        
        for dim in [0, 1]:
            logging.debug(f"--- Checking Stability: Dimension {dim} ---")
            dim_stable = True  # Assume stable until proven otherwise
            max_dist_found = 0.0  # For dim 1 logging
            num_finite_points_history = []  # For dim 0 check
            
            for i in range(self.persistence_history_length):
                pd_to_check = recent_diagrams[-(i + 1)]  # Iterate from latest to oldest
                pd_dim_intervals = pd_to_check.get(dim, [])
                
                # --- Specific check for Dim 0 ---
                if dim == 0:
                    points_dim0 = self._get_diagram_points(pd_dim_intervals)
                    num_finite_points = points_dim0.shape[0]
                    num_finite_points_history.append(num_finite_points)
                    logging.debug(f"    Dim 0: History point -{i}: {num_finite_points} finite points")
                
                # --- Standard Bottleneck check for Dim 1 ---
                elif dim == 1:
                    # Compare latest with previous ones in the window
                    if i > 0:  # Only compare if not the first (latest) diagram
                        latest_pd_dim1 = recent_diagrams[-1].get(dim, [])
                        latest_points_dim1 = self._get_diagram_points(latest_pd_dim1)
                        
                        prev_pd_dim1 = recent_diagrams[-(i + 1)].get(dim, [])
                        prev_points_dim1 = self._get_diagram_points(prev_pd_dim1)
                        
                        if latest_points_dim1.shape[0] == 0 and prev_points_dim1.shape[0] == 0:
                            distance = 0.0
                        elif latest_points_dim1.shape[0] == 0 or prev_points_dim1.shape[0] == 0:
                            distance = float('inf')
                        else:
                            try:
                                distance = bottleneck_distance(latest_points_dim1, prev_points_dim1)
                            except Exception as e:
                                logging.error(f"Error calculating Bottleneck distance (Dim 1): {e}")
                                distance = float('inf')
                        
                        max_dist_found = max(max_dist_found, distance if distance != float('inf') else max_dist_found)
                        
                        if distance > self.stability_threshold:
                            dim_stable = False
                            logging.debug(
                                f"    Stability check dim 1: Dist({len(history) - 1} vs {len(history) - 1 - i - 1}) = {distance:.4f} > Threshold({self.stability_threshold}) -> INSTABILITY.")
                            # No need to check further comparisons for dim 1 if one failed
                            break
                        else:
                            logging.debug(
                                f"    Stability check dim 1: Dist({len(history) - 1} vs {len(history) - 1 - i - 1}) = {distance:.4f} <= Threshold({self.stability_threshold}) -> OK.")
            
            # --- Evaluate Stability for Dimension ---
            if dim == 0:
                # Check if the number of finite points has stabilized (low variance)
                if len(num_finite_points_history) == self.persistence_history_length:
                    variance = np.var(num_finite_points_history)
                    # Define a threshold for variance, e.g., allow variance of 1 or 2 points
                    variance_threshold = 1.5  # Allow some fluctuation
                    if variance > variance_threshold:
                        dim_stable = False
                        logging.debug(
                            f"  Stability check dim 0 result: UNSTABLE. Variance of finite points ({variance:.2f}) > Threshold ({variance_threshold}). Points: {num_finite_points_history}")
                    else:
                        dim_stable = True
                        logging.debug(
                            f"  Stability check dim 0 result: STABLE. Variance of finite points ({variance:.2f}) <= Threshold ({variance_threshold}). Points: {num_finite_points_history}")
                else:
                    dim_stable = False  # Not enough history yet
            
            elif dim == 1:
                logging.debug(
                    f"  Stability check dim 1 result: {'Stable' if dim_stable else 'Unstable'}. Max distance found: {max_dist_found:.4f}")
            
            if not dim_stable:
                overall_stable = False
        
        logging.debug(f"--- Overall Stability Result: {'Stable' if overall_stable else 'Unstable'} ---")
        return overall_stable
    
    def _symbolize_cluster(self, nodes_to_symbolize: FrozenSet[Node]):
        """Assigns an ID to a stable cluster and updates the graph."""
        
        original_symbols = frozenset(s for s in nodes_to_symbolize if isinstance(s, str))
        
        # Double check size based on *currently unassigned* original symbols
        unassigned_original_symbols = frozenset(s for s in original_symbols if s not in self.symbol_to_cluster)
        
        if len(unassigned_original_symbols) < self.min_cluster_size:
            logging.warning(
                f"Cluster {nodes_to_symbolize} has only {len(unassigned_original_symbols)} unassigned original symbols (min: {self.min_cluster_size}), not symbolizing.")
            return
        
        new_id = self.next_cluster_id
        self.next_cluster_id += 1
        logging.info(f"--- SYMBOLIZING CLUSTER --- ID: {new_id}, Members: {nodes_to_symbolize}")
        logging.info(f"Original symbols in cluster: {original_symbols}")
        
        self.cluster_ids.add(new_id)
        # Use only the original symbols for the definition stored
        self.cluster_definitions[new_id] = original_symbols
        
        # --- Update Mappings and Graph ---
        # Counters for the new ID's connections
        new_id_outgoing_counts = Counter()
        new_id_total_occ = 0
        
        nodes_processed_for_symbol_mapping = set()
        
        for node in nodes_to_symbolize:
            # Add node's total occurrences to the new ID
            new_id_total_occ += self.node_total_occurrences.pop(node, 0)  # Use pop to remove
            
            # Update symbol -> cluster mapping for original symbols within the cluster
            if isinstance(node, str) and node not in nodes_processed_for_symbol_mapping:
                if node in self.symbol_to_cluster:
                    logging.error(
                        f"Symbol {node} is already in cluster {self.symbol_to_cluster[node]}, but is part of new cluster {new_id}. This indicates overlapping clusters or logic error!")
                    # Decide on resolution strategy: keep old mapping? force new? merge?
                    # For now, let's log error and potentially overwrite (check impact)
                self.symbol_to_cluster[node] = new_id
                self.symbols.discard(node)  # Remove from active individual symbols
                nodes_processed_for_symbol_mapping.add(node)
            
            # Aggregate co-occurrence counts for the new ID from this node's perspective
            if node in self.cooccurrence_counts:
                node_neighbors = self.cooccurrence_counts.pop(node)  # Get & remove node's outgoing counts
                for neighbor, count in node_neighbors.items():
                    if neighbor in nodes_to_symbolize:
                        # Internal connection - ignore for the new ID's *external* counts
                        pass
                    else:
                        # External connection - needs rewiring
                        # Find the highest representation of the neighbor
                        neighbor_repr = self._get_node_representation(neighbor) if isinstance(neighbor,
                                                                                              str) else neighbor
                        
                        # Add count towards the new ID
                        new_id_outgoing_counts[neighbor_repr] += count
                        
                        # Update the neighbor's perspective: replace 'node' with 'new_id'
                        if neighbor in self.cooccurrence_counts:
                            neighbor_internal_counts = self.cooccurrence_counts[neighbor]
                            # Add count towards new_id, remove count towards old node
                            current_count_to_id = neighbor_internal_counts.get(new_id, 0)
                            neighbor_internal_counts[new_id] = current_count_to_id + count
                            neighbor_internal_counts.pop(node, None)  # Remove old connection safely
                        else:
                            # This case might happen if neighbor itself was just symbolized
                            logging.warning(
                                f"Neighbor {neighbor} not found in cooccurrence_counts during symbolization of {new_id}.")
        
        # Add the new cluster ID to the graph structures
        self.cooccurrence_counts[new_id] = new_id_outgoing_counts
        self.node_total_occurrences[new_id] = new_id_total_occ
        
        # Add incoming counts from external nodes to the new ID
        for external_node, counts in self.cooccurrence_counts.items():
            if external_node != new_id:  # Avoid self processing
                if new_id in counts:  # Check if connection already added from other side
                    self.cooccurrence_counts[new_id][external_node] = counts[new_id]  # Ensure symmetry
        
        logging.info(f"--- SYMBOLIZATION COMPLETE for ID {new_id} ---")
    
    def _detect_components(self, nodes_to_consider: Optional[Set[Node]] = None) -> List[FrozenSet[Node]]:
        """
        Finds connected components in the graph of co-occurrence counts.
        Uses BFS. Considers nodes with counts >= min_cooccurrence_for_ph as connected.
        """
        # If nodes_to_consider is None, use all currently active nodes (symbols + IDs)
        if nodes_to_consider is None:
            # Make sure to include symbols even if not in cooccurrence_counts keys yet
            active_nodes = (self.symbols | self.cluster_ids | set(self.cooccurrence_counts.keys())) - set(
                self.symbol_to_cluster.keys())
            # Also add neighbors to ensure connectivity is fully explored
            neighbors_of_active = set()
            for n in list(active_nodes):  # Use list copy for safe iteration
                if n in self.cooccurrence_counts:
                    neighbors_of_active.update(self.cooccurrence_counts[n].keys())
            active_nodes.update(neighbors_of_active)
            nodes_to_consider = active_nodes
        
        else:
            # Filter provided nodes to only include active symbols or existing IDs
            # And ensure they exist in the graph context
            filtered_nodes = set()
            queue = list(nodes_to_consider)
            visited_for_filtering = set()
            while queue:
                n = queue.pop(0)
                if n in visited_for_filtering:
                    continue
                visited_for_filtering.add(n)
                
                is_active = isinstance(n, int) or n in self.symbols
                if is_active:
                    filtered_nodes.add(n)
                    # Add neighbors too, to ensure components are fully captured
                    if n in self.cooccurrence_counts:
                        for neighbor in self.cooccurrence_counts[n]:
                            if neighbor not in visited_for_filtering:
                                queue.append(neighbor)
            nodes_to_consider = filtered_nodes
        
        visited = set()
        components = []
        
        sorted_nodes_to_consider = sorted(list(nodes_to_consider), key=lambda x: str(x))
        
        for start_node in sorted_nodes_to_consider:
            if start_node not in visited:
                component_nodes = set()
                queue = [start_node]
                visited.add(start_node)
                
                while queue:
                    current_node = queue.pop(0)
                    component_nodes.add(current_node)
                    
                    # Check neighbors based on co-occurrence counts >= threshold
                    if current_node in self.cooccurrence_counts:
                        for neighbor, count in self.cooccurrence_counts[current_node].items():
                            # Neighbor must be relevant and edge strong enough
                            if neighbor in nodes_to_consider and neighbor not in visited \
                                    and count >= self.min_cooccurrence_for_ph:
                                visited.add(neighbor)
                                queue.append(neighbor)
                
                # Add the found component
                if component_nodes:  # Avoid empty components
                    components.append(frozenset(component_nodes))
                    logging.debug(f"Detected component: {component_nodes}")
        
        return components
    
    # --- UPDATED: Heuristic approach focusing on largest symbol component ---
    def _check_for_new_clusters(self, updated_nodes: Set[Node]):
        """
        Checks components involving updated nodes for potential clustering and stability.
        Uses a heuristic: Tracks stability only for the largest component containing
        a sufficient number of original (non-symbolized) symbols.
        """
        potential_components = self._detect_components(updated_nodes)
        
        # --- Heuristic: Find the largest component meeting criteria ---
        largest_symbol_component: Optional[FrozenSet[Node]] = None
        max_original_symbols = 0
        
        for component in potential_components:
            original_symbols_in_component = {s for s in component if
                                             isinstance(s, str) and s not in self.symbol_to_cluster}
            
            # Check if it meets the minimum size based on unassigned original symbols
            if len(original_symbols_in_component) >= self.min_cluster_size:
                # Check if it's larger than the current largest found
                if len(original_symbols_in_component) > max_original_symbols:
                    # Check it's not fully subsumed by an *existing* cluster ID (should be rare if logic is right)
                    is_subsumed = False
                    if len(self.cluster_definitions) > 0:  # Only check if clusters exist
                        component_original_symbols = frozenset(s for s in component if isinstance(s, str))
                        if component_original_symbols and any(component_original_symbols.issubset(defn) for defn in
                                                              self.cluster_definitions.values()):
                            is_subsumed = True
                            logging.debug(
                                f"Component {component} seems already part of an existing cluster definition, skipping as main candidate.")
                    
                    if not is_subsumed:
                        max_original_symbols = len(original_symbols_in_component)
                        largest_symbol_component = component
        
        # --- Process the largest candidate found ---
        if largest_symbol_component:
            logging.debug(
                f"Largest candidate component identified: {largest_symbol_component} (Size: {len(largest_symbol_component)}, Original Symbols: {max_original_symbols})")
            candidate_key = "main_candidate"  # Use a fixed key for tracking this heuristic target
            
            # Ensure the tracking entry exists
            if candidate_key not in self.potential_clusters:
                logging.info(f"Initializing tracking for candidate '{candidate_key}'")
                self.potential_clusters[candidate_key] = {
                    'persistence_history': [],
                    'stability_checks_passed': 0,
                    'current_members': largest_symbol_component
                }
            else:
                # Update members if changed
                self.potential_clusters[candidate_key]['current_members'] = largest_symbol_component
            
            # Calculate persistence for the *current* largest component
            simplex_tree = self._build_simplex_tree(largest_symbol_component)
            if simplex_tree:
                logging.debug(f"Calculating PH for candidate '{candidate_key}'...")
                current_persistence = self._calculate_persistence(simplex_tree)
                
                # Append PH to the history associated with the fixed key
                history_list = self.potential_clusters[candidate_key]['persistence_history']
                history_list.append(current_persistence)
                
                # Limit history
                max_hist = max(10, self.persistence_history_length * 2)
                if len(history_list) > max_hist:
                    history_list.pop(0)
                
                logging.debug(f"Candidate '{candidate_key}' history length: {len(history_list)}")
                
                # Check stability using the history under the fixed key
                if len(history_list) >= self.persistence_history_length:
                    logging.debug(f"Checking stability for candidate '{candidate_key}'...")
                    is_stable = self._check_stability_on_history(history_list)
                    
                    if is_stable:
                        self.potential_clusters[candidate_key]['stability_checks_passed'] += 1
                        logging.info(
                            f"Candidate '{candidate_key}' PASSED stability check {self.potential_clusters[candidate_key]['stability_checks_passed']} (requires {self.required_stability_passes}).")
                    else:
                        # Reset counter if stability fails
                        self.potential_clusters[candidate_key]['stability_checks_passed'] = 0
                        logging.info(f"Candidate '{candidate_key}' FAILED stability check.")
                    
                    # Check if enough passes achieved
                    if self.potential_clusters[candidate_key][
                        'stability_checks_passed'] >= self.required_stability_passes:
                        members_to_symbolize = self.potential_clusters[candidate_key]['current_members']
                        # Final check on original symbols before symbolizing
                        final_original_symbols = {s for s in members_to_symbolize if
                                                  isinstance(s, str) and s not in self.symbol_to_cluster}
                        
                        if len(final_original_symbols) >= self.min_cluster_size:
                            # SYMBOLIZE!
                            self._symbolize_cluster(members_to_symbolize)
                            # Remove the tracking entry after successful symbolization
                            if candidate_key in self.potential_clusters:
                                del self.potential_clusters[candidate_key]
                        else:
                            logging.warning(
                                f"Stable candidate '{candidate_key}' became too small before symbolization ({len(final_original_symbols)} unassigned original symbols). Resetting stability pass count.")
                            # Reset passes, keep tracking maybe? Or remove? Let's reset.
                            self.potential_clusters[candidate_key]['stability_checks_passed'] = 0
                    # --- End Symbolization Check ---
                else:
                    logging.debug(
                        f"Candidate '{candidate_key}' needs {self.persistence_history_length} PH calculations for stability check (has {len(history_list)}).")
            else:
                logging.debug(f"Could not build simplex tree for candidate '{candidate_key}'. Skipping PH.")
        else:
            logging.debug("No suitable largest symbol component found in this step.")
    
    def get_state(self) -> Dict:
        """Returns the current state of the clusterer, ready for JSON serialization."""
        serializable_potential_clusters = {}
        for key, data in self.potential_clusters.items():
            # Handle heuristic key (should be the only type now)
            serializable_potential_clusters[key] = {
                "current_members": sorted(list(data.get('current_members', set())), key=str),
                "history_len": len(data.get('persistence_history', [])),
                "stability_passes": data.get('stability_checks_passed', 0)
            }
        
        # Convert cooccurrence keys/values
        serializable_cooc = {}
        for node1, neighbors in self.cooccurrence_counts.items():
            str_node1 = str(node1)
            serializable_cooc[str_node1] = {}
            for node2, count in neighbors.items():
                serializable_cooc[str_node1][str(node2)] = count
        
        return {
            "symbols": sorted(list(self.symbols)),
            "cluster_ids": sorted(list(self.cluster_ids)),
            "cluster_definitions": {k: sorted(list(v), key=str) for k, v in self.cluster_definitions.items()},
            "symbol_to_cluster": self.symbol_to_cluster,
            "cooccurrence_counts": serializable_cooc,
            "node_total_occurrences": {str(node): count for node, count in self.node_total_occurrences.items()},
            "potential_clusters": serializable_potential_clusters,
            "next_cluster_id": self.next_cluster_id
        }