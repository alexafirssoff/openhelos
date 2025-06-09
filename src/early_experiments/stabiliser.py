import math
import numpy as np
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from scipy.spatial import distance as sp_distance
from namer import Namer
from binstorage import BinaryStorage


# --- Hyperparameters ---
@dataclass
class StabilizerHyperParams:
    """ Hyperparameters for the Stabilizer """
    # --- EMA Parameters ---
    ema_beta_symbol: float = 0.3  # Smoothing coefficient for P(Cluster|Symbol)
    ema_beta_transition: float = 0.3  # Smoothing coefficient for P(Next|Prev) (slower?)
    # --- Stability Parameters ---
    min_freq_for_stable: int = 3  # Min. cluster frequency for a symbol to be considered stable
    min_prob_for_stable: float = 0.4  # Min. EMA probability for selecting a stable ID
    # --- Clustering Parameters ---
    # Distance threshold for creating a new cluster (fraction of average norm?)
    # Or just maximum distance? Depends on the metric.
    # For cosine DISTANCE (1-similarity): threshold could be 0.3-0.5?
    # For Euclidean: depends on the scale of activations.
    # For now, using Cosine distance:
    clustering_threshold_cosine: float = 0.4  # Max. cosine distance for assigning to a cluster
    # Centroid update rate (alpha = 1 / (count+1)) is used by default
    # min_samples_for_centroid: int = 1 # Minimum number of samples for a centroid to exist
    current_cluster_bonus: float = 0.4


# --- Main Class ---
class Stabilizer:
    def __init__(
            self,
            name: int,
            manager_storage_path: Path,
            num_columns: int,
            hyperparameters: StabilizerHyperParams = StabilizerHyperParams()
    ):
        self.name = name
        self.stabilizer_storage_path = manager_storage_path / f"stabilizer_{name}"  # Folder for stabilizer state
        self.stabilizer_storage_path.mkdir(parents=True, exist_ok=True)
        self.hyper_params = hyperparameters
        self.num_columns = num_columns  # Needed for signature dimensionality
        self.centroid_origins: Dict[int, int] = {}
        
        # --- Internal State ---
        # Centroid Dictionary: {ClusterID: np.array(shape=(num_columns,))}
        self.centroids: Dict[int, np.ndarray] = {}
        # Sample Counts for Centroids: {ClusterID: int}
        self.centroid_counts: Dict[int, int] = defaultdict(int)
        # Map "Symbol -> {ClusterID -> Statistics}"
        self.symbol_cluster_stats: Dict[Any, Dict[int, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {'count': 0, 'ema_prob': 0.0}))
        # Transition Map "PrevClusterID -> {NextClusterID -> Statistics}"
        self.transition_stats: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {'count': 0, 'ema_prob': 0.0}))
        # Previous stable ID (for updating transitions)
        self.previous_stable_id: Optional[int] = None
        # Next ID for a new cluster
        self.next_cluster_id: int = 0
        
        # --- Namer for ID Sets (Shared with ColumnManager) ---
        # Get path to ColumnManager's namers folder
        namer_path = manager_storage_path / 'namers'
        namer_path.mkdir(parents=True, exist_ok=True)  # Create if needed
        self.column_set_namer = Namer(1, namer_path)  # Use ID=1
        
        # --- Load State ---
        self.state_file_path = self.stabilizer_storage_path / f"stabilizer_{self.name}_state.bin"
        self._load_state()
    
    def _serialize_state(self) -> Dict[str, Any]:
        """ Converts state to a serialisable format. """
        # Convert numpy arrays to lists
        serializable_centroids = {k: v.tolist() for k, v in self.centroids.items()}
        # Convert symbol_cluster_stats keys if they are not primitives
        serializable_symbol_stats = {}
        for symbol, sets_dict in self.symbol_cluster_stats.items():
            # Use repr() for non-primitive keys
            key = symbol if isinstance(symbol, (str, int, float, bool, tuple)) else repr(symbol)
            serializable_symbol_stats[key] = {str(k): v for k, v in sets_dict.items()}
        # Convert transition_stats keys
        serializable_transition_stats = {}
        for prev_id, next_dict in self.transition_stats.items():
            serializable_transition_stats[str(prev_id)] = {str(k): v for k, v in next_dict.items()}
        
        return {
            'centroids': serializable_centroids,
            'centroid_counts': dict(self.centroid_counts),  # Convert defaultdict
            'symbol_cluster_stats': serializable_symbol_stats,
            'transition_stats': serializable_transition_stats,
            'previous_stable_id': self.previous_stable_id,
            'next_cluster_id': self.next_cluster_id,
            'centroid_origins': self.centroid_origins,
        }
    
    def _deserialize_state(self, loaded_data: Dict[str, Any]):
        """ Restores state from deserialised data. """
        loaded_centroids = loaded_data.get('centroids', {})
        self.centroids = {int(k): np.array(v) for k, v in loaded_centroids.items()}  # Restore numpy
        
        self.centroid_counts = defaultdict(int, loaded_data.get('centroid_counts', {}))
        
        # Restore symbol_cluster_stats (Reverse conversion of keys is more complex)
        # For simplicity, we will use strings as keys if the symbol was not a primitive
        loaded_symbol_stats_raw = loaded_data.get('symbol_to_set_stats', {})
        self.symbol_cluster_stats.clear()
        for symbol_key, sets_dict_raw in loaded_symbol_stats_raw.items():
            # Try to restore primitives, otherwise leave as string
            try:
                symbol = eval(symbol_key) if symbol_key.startswith(('(', '[', '{')) else symbol_key
            except:
                symbol = symbol_key
            self.symbol_cluster_stats[symbol] = defaultdict(lambda: {'count': 0, 'ema_prob': 0.0})
            for set_id_str, stats_dict in sets_dict_raw.items():
                try:
                    self.symbol_cluster_stats[symbol][int(set_id_str)] = stats_dict
                except ValueError:
                    pass
        
        # Restore transition_stats
        loaded_transition_stats_raw = loaded_data.get('transition_stats', {})
        self.transition_stats.clear()
        for prev_id_str, next_dict_raw in loaded_transition_stats_raw.items():
            try:
                prev_id = int(prev_id_str)
                self.transition_stats[prev_id] = defaultdict(lambda: {'count': 0, 'ema_prob': 0.0})
                for next_id_str, stats_dict in next_dict_raw.items():
                    try:
                        self.transition_stats[prev_id][int(next_id_str)] = stats_dict
                    except ValueError:
                        pass
            except ValueError:
                pass
        
        self.centroid_origins = {int(k): int(v) for k, v in loaded_data.get('centroid_origins', {}).items()}
        self.previous_stable_id = loaded_data.get('previous_stable_id')
        self.next_cluster_id = loaded_data.get('next_cluster_id', 0)
    
    def _load_state(self):
        if self.state_file_path.exists():
            try:
                storage = BinaryStorage(self.state_file_path, {})
                self._deserialize_state(storage.data)
                print(f"Stabilizer '{self.name}': State loaded. Next Cluster ID: {self.next_cluster_id}")
            except Exception as e:
                print(f"ERROR Stabilizer '{self.name}': Failed to load state: {e}");
                traceback.print_exc()
        else:
            print(f"Stabilizer '{self.name}': No state file found, starting fresh.")
    
    def dump(self):
        try:
            serializable_data = self._serialize_state()
            storage = BinaryStorage(self.state_file_path, {})
            storage.data = serializable_data
            storage.dump()
            # print(f"Stabilizer '{self.name}': State saved.")
        except Exception as e:
            print(f"ERROR Stabilizer '{self.name}': Failed to save state: {e}");
            traceback.print_exc()
    
    def _find_or_create_cluster(self, raw_signature: np.ndarray, context_col_id: int) -> int:
        """
        Finds the nearest cluster for the signature or creates a new one.
        Updates the centroid of the chosen cluster.
        Returns the cluster ID.
        """
        if not isinstance(raw_signature, np.ndarray):
            # Convert list to numpy if necessary
            try:
                raw_signature = np.array(raw_signature, dtype=float)
            except:
                print("ERROR: Invalid raw_signature format.");
                return -1  # Error ID
        # Normalise signature for cosine distance? L2 normalisation
        norm = np.linalg.norm(raw_signature)
        if norm < 0.2:  # New hyperparameter
            return -1  # Or WEAK_CLUSTER_ID
        signature_norm = raw_signature / (norm + 1e-9)  # Add epsilon for stability
        
        # +++ START OF CLUSTERING DEBUGGING BLOCK +++
        # print(f"    STAB DEBUG (_find_or_create): --- Processing Signature ---")
        # print(f"      Norm Signature: {np.array2string(signature_norm, precision=3, suppress_small=True)}")
        top_indices_norm = np.argsort(signature_norm)[-3:][::-1]
        top_values_norm = signature_norm[top_indices_norm]
        # print(f"      Top 3 Norm FS: { {k: f'{v:.3f}' for k, v in zip(top_indices_norm, top_values_norm)} }")
        # +++ END OF CLUSTERING DEBUGGING BLOCK +++
        
        if not self.centroids:  # If this is the first signature
            new_id = self.next_cluster_id
            self.centroids[new_id] = signature_norm  # Save normalised
            self.centroid_counts[new_id] = 1
            self.next_cluster_id += 1
            self.centroid_origins[new_id] = context_col_id
            # +++ ADD PRINTING OF FIRST CENTROID +++
            # print(f"    STAB DEBUG (_find_or_create): Created FIRST Cluster {new_id}")
            # print(
            #     f"      Centroid {new_id}: {np.array2string(self.centroids[new_id], precision=3, suppress_small=True)}")
            # +++ END OF PRINTING +++
            # print(f"      DEBUG Cluster: Created first cluster {new_id}")
            return new_id
        
        # Find the nearest centroid
        best_cluster_id = -1
        # Use cosine DISTANCE (1 - similarity)
        min_dist = float('inf')
        distances_to_centroids = {}
        
        for cluster_id, centroid in self.centroids.items():
            # Centroids are ALREADY normalised when saved/updated
            # dist = sp_distance.euclidean(signature_norm, centroid)
            try:
                # dist = sp_distance.cosine(signature_norm, centroid)
                # distances_to_centroids[cluster_id] = dist  # Save distance
                dist = sp_distance.euclidean(signature_norm, centroid)
                distances_to_centroids[cluster_id] = dist
                # +++ START OF DISTANCE DEBUGGING BLOCK +++
                # Print distance to key clusters (0, 1, 2, 3...)
                clusters_to_debug = [0, 1, 2, 3]  # Change as needed
                if cluster_id in clusters_to_debug:
                    # print(f"      Dist to Centroid {cluster_id}: {dist:.4f}")
                    # If you need to see the centroid itself for comparison:
                    # print(f"        Centroid {cluster_id}: {np.array2string(centroid, precision=3, suppress_small=True)}")
                    # +++ END OF DISTANCE DEBUGGING BLOCK +++
                    ...
            except Exception as e:
                # print(f"      WARN: Cosine distance error for C{cluster_id}: {e}. Sig={signature_norm}, Cent={centroid}")
                dist = float('inf')  # Ignore this centroid
            
            if dist < min_dist:
                min_dist = dist
                best_cluster_id = cluster_id
        
        threshold = self.hyper_params.clustering_threshold_cosine  # Use hyperparameter
        action_taken = "No action"
        
        # Decide whether to create a new cluster
        if best_cluster_id == -1 or min_dist > self.hyper_params.clustering_threshold_cosine:
            # Create new
            assigned_cluster_id = self.next_cluster_id
            self.centroids[assigned_cluster_id] = signature_norm
            self.centroid_counts[assigned_cluster_id] = 1
            self.centroid_origins[assigned_cluster_id] = context_col_id
            self.next_cluster_id += 1
            action_taken = f"CREATED New Cluster {assigned_cluster_id} (Origin Col {context_col_id})"
            # print(f"      DEBUG Cluster: Created new cluster {assigned_cluster_id} (MinDist={min_dist:.3f} > Thr)")
        else:
            # Assign to existing
            assigned_cluster_id = best_cluster_id
            origin_col_id = self.centroid_origins.get(assigned_cluster_id, -1)
            
            # --- CONTEXT CHECK BEFORE UPDATING ---
            if origin_col_id == -1 or origin_col_id == context_col_id:
                # Contexts match or origin unknown -> UPDATE
                count = self.centroid_counts.get(assigned_cluster_id, 0)
                self.centroid_counts[assigned_cluster_id] = count + 1  # Update counter here
                alpha = 1.0 / (count + 1.0)
                # Ensure centroid exists before updating
                if assigned_cluster_id in self.centroids:
                    updated_centroid = (1.0 - alpha) * self.centroids[assigned_cluster_id] + alpha * signature_norm
                    updated_norm = np.linalg.norm(updated_centroid)
                    self.centroids[assigned_cluster_id] = updated_centroid / (updated_norm + 1e-9)
                    action_taken = f"Assigned to C{assigned_cluster_id} & UPDATED Centroid (Context Match: Cur={context_col_id}, Orig={origin_col_id})"
                else:
                    # This should not happen, but just in case
                    self.centroids[assigned_cluster_id] = signature_norm  # Simply write the current one
                    action_taken = f"Assigned to C{assigned_cluster_id} & INITIALIZED Centroid (Prev Missing?) (Context Match: Cur={context_col_id}, Orig={origin_col_id})"
            
            else:
                # Contexts DO NOT match -> DO NOT UPDATE CENTROID
                # Simply assign ID, DO NOT increment counter (or increment a separate "visits" counter?)
                action_taken = f"Assigned to C{assigned_cluster_id} & SKIPPED Update (Context Mismatch: Cur={context_col_id}, Orig={origin_col_id})"
                # If we do not update the counter, then alpha at the next "correct" update will be as needed.
                # If the counter is updated, alpha will decrease faster. For now, do not update.
        
        # ... (Debug print with action_taken) ...
        # print(f"      Action Taken: {action_taken}")
        
        return assigned_cluster_id
    
    def _update_symbol_cluster_stats(self, symbol: Any, assigned_cluster_id: int):
        """ Updates statistics for the (symbol, cluster ID) pair using EMA. """
        beta = self.hyper_params.ema_beta_symbol
        symbol_stats = self.symbol_cluster_stats[symbol]
        total_prev_prob = sum(stats.get('ema_prob', 0.0) for stats in symbol_stats.values())  # Sum before update
        
        stats_observed = symbol_stats[assigned_cluster_id]
        stats_observed['count'] = stats_observed.get('count', 0) + 1
        # Update EMA only if total probability was not too small
        if total_prev_prob > 1e-9:
            stats_observed['ema_prob'] = (1 - beta) * stats_observed.get('ema_prob', 0.0) + beta * 1.0
            # Weaken others
            for set_id, stats in symbol_stats.items():
                if set_id != assigned_cluster_id:
                    stats['ema_prob'] = (1 - beta) * stats.get('ema_prob', 0.0)
        else:  # If this is the first observation for the symbol or probabilities were zeroed out
            for set_id in symbol_stats.keys():  # Reset all
                symbol_stats[set_id]['ema_prob'] = 0.0
            stats_observed['ema_prob'] = 1.0  # Set 1 for the current one
        
        # Normalise probabilities
        total_prob = sum(stats.get('ema_prob', 0.0) for stats in symbol_stats.values())
        if total_prob > 1e-9:
            for set_id in symbol_stats.keys():
                symbol_stats[set_id]['ema_prob'] /= total_prob
        # else: leave 1.0 for the single observed one
    
    def _update_transition_stats(self, current_stable_cluster_id: int):
        """ Updates transition statistics using EMA. """
        beta = self.hyper_params.ema_beta_transition
        if self.previous_stable_id is not None:
            transitions_from_prev = self.transition_stats[self.previous_stable_id]
            total_prev_prob = sum(stats.get('ema_prob', 0.0) for stats in transitions_from_prev.values())
            
            stats_observed = transitions_from_prev[current_stable_cluster_id]
            stats_observed['count'] = stats_observed.get('count', 0) + 1
            
            if total_prev_prob > 1e-9:
                stats_observed['ema_prob'] = (1 - beta) * stats_observed.get('ema_prob', 0.0) + beta * 1.0
                for next_id, stats in transitions_from_prev.items():
                    if next_id != current_stable_cluster_id:
                        stats['ema_prob'] = (1 - beta) * stats.get('ema_prob', 0.0)
            else:  # First transition from this state
                for next_id in transitions_from_prev.keys():
                    transitions_from_prev[next_id]['ema_prob'] = 0.0
                stats_observed['ema_prob'] = 1.0
            
            # Normalise
            total_prob = sum(stats.get('ema_prob', 0.0) for stats in transitions_from_prev.values())
            if total_prob > 1e-9:
                for next_id in transitions_from_prev.keys():
                    transitions_from_prev[next_id]['ema_prob'] /= total_prob
        
        self.previous_stable_id = current_stable_cluster_id  # Update for the next step
    
    def get_stable_cluster_id(
            self,
            symbol: Any,
            raw_signature_vector: List[float],
            is_window_novel: bool,  # New argument
            context_col_id: int
    ) -> int:
        """
        Clusters the raw signature, updates statistics,
        and returns a stable cluster ID, trusting the current observation during novelty.
        """
        # 1. Cluster -> current_cluster_id
        try:
            raw_signature_np = np.array(raw_signature_vector, dtype=float)
        except:
            print(f"ERROR Stabilizer: Cannot convert raw_signature for '{symbol}'.");
            return -1
        current_cluster_id = self._find_or_create_cluster(raw_signature_np, context_col_id)
        if current_cluster_id == -1:
            return -1
        
        # 2. Update symbol -> cluster statistics
        self._update_symbol_cluster_stats(symbol, current_cluster_id)
        
        # 3. Select Stable Cluster ID
        stable_cluster_id = -1
        symbol_stats = self.symbol_cluster_stats.get(symbol)
        
        # --- NEW LOGIC: CONSIDER WINDOW NOVELTY ---
        if is_window_novel:
            # If CM signals window novelty, trust the current cluster
            stable_cluster_id = current_cluster_id
            # print(f"      DEBUG Stabilizer: Window Novelty HIGH. Using CURRENT cluster {stable_cluster_id} for '{symbol}'.")
        # --- OLD LOGIC (for NON-NOVEL windows, with bonus to current) ---
        elif symbol_stats:
            belief: Dict[int, float] = {
                cid: stats.get('ema_prob', 0.0)
                for cid, stats in symbol_stats.items()
            }
            if belief:
                # Add bonus to belief in the current cluster
                current_bonus = self.hyper_params.current_cluster_bonus
                belief[current_cluster_id] = belief.get(current_cluster_id, 0.0) + current_bonus
                # Find cluster with maximum adjusted "belief"
                try:
                    best_stable_candidate = max(belief, key=lambda k: belief.get(k, -math.inf))
                    stable_cluster_id = best_stable_candidate
                    # print(f"      DEBUG Stabilizer: Stable ID {stable_cluster_id} chosen for \'{symbol}\' (AdjBelief: {belief.get(stable_cluster_id, 0.0):.3f}, Current: {current_cluster_id})")
                except ValueError:
                    print(f"      WARN Stabilizer: Could not determine max belief for symbol '{symbol}'.")
                    stable_cluster_id = current_cluster_id  # Fallback to current
        
        # If still not determined (no statistics), use current
        if stable_cluster_id == -1:
            stable_cluster_id = current_cluster_id
            # print(f\"      DEBUG Stabilizer: Using current cluster ID {stable_cluster_id} for \'{symbol}\' (default).\")
        
        # 4. Update transition statistics (based on chosen stable_cluster_id)
        self._update_transition_stats(stable_cluster_id)
        
        return stable_cluster_id
    
    def predict_next_distribution(self) -> Dict[int, float]:
        """
        Predicts the probability distribution for the next Stable Cluster ID.
        """
        if self.previous_stable_id is not None and self.previous_stable_id in self.transition_stats:
            predicted_dist = {next_id: stats.get('ema_prob', 0.0)
                              for next_id, stats in self.transition_stats[self.previous_stable_id].items()}
            # print(f"    DEBUG Stabilizer Predict: Dist from {self.previous_stable_id}: { {k:f'{v:.3f}' for k,v in predicted_dist.items()} }")
            return predicted_dist
        else:
            # print(f"    DEBUG Stabilizer Predict: No history for previous state ({self.previous_stable_id}).")
            return {}
    
    def get_expectedness(self, predicted_dist: Dict[int, float], column_id: int) -> float:
        """
        Calculates the "expectedness" of a column based on the predicted
        distribution of cluster IDs, using centroid vectors.
        """
        total_expectedness = 0.0
        if not predicted_dist or column_id < 0 or column_id >= self.num_columns:
            return 0.0  # If no prediction or invalid column ID
        
        # Iterate through predicted clusters and their probabilities
        for cluster_id, prob in predicted_dist.items():
            if prob < 1e-9:
                continue  # Skip almost zero probabilities
            
            # Get cluster representation - its centroid
            centroid_vector = self.centroids.get(cluster_id)
            
            if centroid_vector is not None and len(centroid_vector) == self.num_columns:
                # "Expectedness" contributed by this cluster is the value
                # of the corresponding component in the (normalised) centroid vector,
                # weighted by the prediction probability of this cluster.
                # Centroids are stored normalised (L2), their values are [0, 1] (approximately)
                # or can be in another range if normalisation changes.
                # Ensure the value is within reasonable limits.
                component_value = centroid_vector[column_id]
                # If centroid values are not normalised to [0,1],
                # additional scaling might be needed, or just use as is.
                # For now, assume centroid components reflect the relative
                # "activity" or "importance" of the column for this cluster.
                contribution = max(0.0, component_value)  # Take only non-negative values
                total_expectedness += prob * contribution
            # else: If centroid for cluster_id is not found, it does not contribute
        
        # Limit to [0, 1], as the sum of prob * component_value could theoretically
        # exceed 1 if centroid components > 1 (unlikely with L2 norm).
        final_expectedness = max(0.0, min(1.0, total_expectedness))
        
        # --- Debugging ---
        # if column_id == 0 or column_id == 1: # Print for columns of interest
        #    print(f"    STAB DEBUG (get_expectedness): Col {column_id}, TotalExpectedness={final_expectedness:.4f} from Dist={ {k:f'{v:.2f}' for k,v in predicted_dist.items()} }")
        # ---------------
        
        return final_expectedness
    
    def plot_centroid_heatmap(self, save_path: Optional[Path] = None):
        """
        Builds and displays or saves a heatmap of cluster centroids.

        Each row represents a cluster centroid, each column a column.
        The colour and value of a cell show the average activity (p_single)
        of the corresponding column for that cluster.

        Args:
            save_path: Optional path (Path) for saving the file.
                       If None, the graph is displayed on the screen.
        """
        print(f"Stabilizer '{self.name}': Generating centroid heatmap...")
        
        if not self.centroids:
            print("  WARN: No centroids found to plot.")
            return
        
        try:
            # Get cluster IDs and sort for consistent order
            sorted_cluster_ids = sorted(self.centroids.keys())
            
            # Create centroid matrix: rows - clusters, columns - columns
            centroid_matrix = np.array([self.centroids[cid] for cid in sorted_cluster_ids])
            
            # Check in case the matrix is empty after assembly (unlikely, but still)
            if centroid_matrix.size == 0:
                print("  WARN: Centroid matrix is empty.")
                return
            
            # --- Building the Heatmap ---
            num_clusters = len(sorted_cluster_ids)
            # Adaptive figure size
            fig_height = max(4, num_clusters * 0.6)
            fig_width = max(8, self.num_columns * 0.8)
            
            plt.figure(figsize=(fig_width, fig_height))
            sns.heatmap(
                centroid_matrix,
                annot=True,  # Show values in cells
                fmt=".2f",  # Number format (2 decimal places)
                cmap="viridis",  # Colour scheme (can choose another, e.g., "plasma", "magma")
                linewidths=.5,  # Lines between cells
                linecolor='lightgray',  # Line colour
                cbar=True,  # Show colour bar
                # Axis labels
                yticklabels=[f"C{cid}" for cid in sorted_cluster_ids],  # Cluster IDs
                xticklabels=[f"Col{i}" for i in range(self.num_columns)]  # Column IDs
            )
            plt.title(f"Stabilizer '{self.name}' - Cluster Centroid Heatmap (p_single based)")
            plt.xlabel("Column ID")
            plt.ylabel("Cluster ID")
            plt.xticks(rotation=0)  # Horizontal column labels
            plt.yticks(rotation=0)  # Horizontal cluster labels
            plt.tight_layout()
            
            # --- Display or Save ---
            if save_path:
                try:
                    # Ensure directory exists
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(save_path, dpi=150)  # Save with good resolution
                    print(f"  INFO: Centroid heatmap saved to {save_path}")
                    plt.close()  # Close figure after saving
                except Exception as e_save:
                    print(f"  ERROR: Failed to save heatmap to {save_path}: {e_save}")
                    plt.show()  # Show on screen in case of save error
            else:
                plt.show()  # Display on screen
        
        except Exception as e_plot:
            print(f"  ERROR: Failed to generate centroid heatmap: {e_plot}")
            traceback.print_exc()
            # Attempt to close the figure if it was created
            if 'plt' in locals() and plt.get_fignums():
                plt.close()