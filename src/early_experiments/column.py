from collections import deque

import sliding_window

# --- Helper Functions and Constants (repeated for completeness) ---
EPSILON = 1e-10
FREQ_DAMPING_FACTOR = 5.0
ALPHA_SINGLE_STABILITY = 0.1
MAX_HISTORY_S = 100
ALPHA_FINAL_FS = 0.5  # Weight of FS_from_pairs in the final FS of an element
K_FAMILIARITY = 1.0  # Coefficient in exp(-k*F) for FS


def get_bigrams_zip(data: list) -> list:
    """Forms a list of bigrams from the input list."""
    if len(data) <= 1:
        return []
    return list(zip(data, data[1:]))


def pair(k1, k2) -> int:
    """Cantor pairing function."""
    k1 = int(k1)
    k2 = int(k2)
    if k1 < 0 or k2 < 0:
        raise ValueError(f"Cantor pairing needs non-negative integers, got {k1}, {k2}")
    return int(0.5 * (k1 + k2) * (k1 + k2 + 1) + k2)


def calculate_familiarity_score(F, k=K_FAMILIARITY) -> float:
    """Calculates the Familiarity Score (FS) from Free Energy (F)."""
    if F < 0 or math.isnan(F):
        return 0.0
    try:
        if math.isinf(F):
            return 0.0
        fs = math.exp(-k * F)
        return fs
    except OverflowError:
        return 0.0
    except Exception:
        return 0.0


# --- End of Helper Functions ---


import math
import shutil
import traceback  # Import traceback for use in except
from collections import defaultdict

from simpicial_neuron import SimplicialNeuron
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set, Optional  # Add Optional

# Assuming binstorage and necessary functions are present
try:
    from binstorage import BinaryStorage
except ImportError:
    print("Warning: binstorage not found. Persistence will not work.")
    
    
    # Add __init__ to the stub for correct load/save operation
    class BinaryStorage:
        def __init__(self, path, default_data): self.path = path; self.data = default_data
        
        def dump(self): pass
        
        def get(self, key, default): return self.data.get(key, default)

try:
    from pairing_function import pair
except ImportError:
    print("Warning: pairing_function not found. Using stub.")
    
    
    def pair(k1, k2):
        return int(k1) * 1000 + int(k2)  # Simple stub

# --- Helper Functions and Constants ---
# Use K_FAMILIARITY from SimplicialNeuron if defined there,
# or define it here as a constant (or pass it in init)
# For example, define it here
K_FAMILIARITY = 0.7  # Example from log


def get_bigrams_zip(data: list) -> list:
    if len(data) <= 1:
        return []
    return list(zip(data, data[1:]))


from familiarity_score import calculate_familiarity_score


# --- End of Helpers ---


class Column:
    def __init__(self, name: int, input_width: int, storage_path: Path):
        """
        Initialises a column of neurons.

        Creates or loads the column's state, including its metadata
        (e.g., `input_width`) and all simplicial neurons it contains.

        Args:
            name: Unique integer ID of the column.
            input_width: Input width (maximum number of elements),
                         determining the number of layers and neurons in the column.
            storage_path: Path to the *parent* directory where the data folders
                          for *all* columns will be stored (e.g., './data/columns').
                          The folder for this column will be `{storage_path}/col_{name}`.
        """
        self.name = name
        if input_width <= 1:
            raise ValueError("input_width must be >= 2 to form at least one layer.")
        
        self.column_storage_path = storage_path / f'col_{name}'
        self.column_storage_path.mkdir(parents=True, exist_ok=True)
        path_to_neurons = self.column_storage_path / 'neurons'
        path_to_neurons.mkdir(parents=True, exist_ok=True)
        
        # --- Loading/Saving Metadata ---
        column_meta_path = self.column_storage_path / 'column_meta.bin'
        # Initialise with default input_width
        initial_meta = {
            'input_width': input_width,
            'max_overall_fs_history_len': 30,
            'overall_fs_history': [],
            'training_steps_count': 0,
            'global_vertex_map': {},
            'next_vertex_id': 0
        }
        try:
            # Load existing or use initial_meta
            self.meta_storage = BinaryStorage(column_meta_path, initial_meta)
            # Load input_width from file if it exists, otherwise use the passed one
            self.input_width = self.meta_storage.data.get('input_width', input_width)
        except Exception as e:
            print(f"Warning Col {self.name}: Error initializing/loading meta storage: {e}")
            # In case of an error, use the passed input_width
            self.input_width = input_width
            # Attempt to create a storage object for subsequent saving
            try:
                self.meta_storage = BinaryStorage(column_meta_path, {'input_width': self.input_width})
            except Exception as e_fallback:
                print(f"ERROR Col {self.name}: Failed to create fallback meta storage: {e_fallback}")
                self.meta_storage = None  # Indicate that metadata cannot be saved
        
        # Ensure we have the input_width attribute
        if not hasattr(self, 'input_width'):
            self.input_width = input_width  # As a last resort
        
        # Save the current width back (if storage is available)
        if self.meta_storage:
            self.meta_storage.data['input_width'] = self.input_width
        
        # --- NEW ATTRIBUTES FOR TRACKING STABILITY ---
        self.max_overall_fs_history_len: int = self.meta_storage.data.get('max_overall_fs_history_len',
                                                                          30)  # Parameter: length of FS history for stability analysis (to be tuned)
        # Loading these attributes (if they were saved)
        # Use get with initial_memory or default
        loaded_history = self.meta_storage.data.get('overall_fs_history', [])  # Load as a list
        # Convert the loaded list to a deque with the required length
        self.overall_fs_history = deque(loaded_history, maxlen=self.max_overall_fs_history_len)
        self.training_steps_count = self.meta_storage.data.get('training_steps_count', 0)
        # --- END OF NEW ATTRIBUTES ---
        
        # --- Creating/Loading Neurons ---
        self.neurons: List[List[SimplicialNeuron]] = []
        self.global_vertex_map = self.meta_storage.data.get('global_vertex_map', dict())
        self.next_vertex_id = self.meta_storage.data.get('next_vertex_id', 0)
        neuron_name_counter = 0
        # Correct calculation of the number of layers
        num_layers_expected = self.input_width - 1
        neuron_global_id_counter = 0  # Local ID counter INSIDE the column
        
        for layer_idx in range(num_layers_expected):  # Layer 0 (top, 1 neuron) .. N-2 (bottom, N-1 neuron)
            neurons_in_this_layer = layer_idx + 1
            current_layer_neurons = []
            for neuron_idx_in_layer in range(neurons_in_this_layer):
                # neuron_global_name = neuron_name_counter
                max_neurons_per_col = (self.input_width - 1) * self.input_width // 2
                neuron_global_name = (max_neurons_per_col * self.name) + neuron_global_id_counter
                # --------------------------------------------
                try:
                    # Attempt to load/create neuron
                    neuron = SimplicialNeuron(neuron_global_name, path_to_neurons)
                except Exception as e:
                    print(f"Warning Col {self.name}: Error loading N {neuron_global_name}, creating new. Error: {e}")
                    neuron_file_path = path_to_neurons / f'{neuron_global_name}.bin'
                    if neuron_file_path.exists():
                        try:
                            neuron_file_path.unlink()
                        except OSError as e_del:
                            print(f"Warning: Could not delete file {neuron_file_path}: {e_del}")
                    # Create a new one, passing bias
                    neuron = SimplicialNeuron(neuron_global_name, path_to_neurons, ordered_z=False)
                current_layer_neurons.append(neuron)
                neuron_name_counter += 1
                neuron_global_id_counter += 1
            # The list index corresponds to the layer number from the top (0 - top)
            self.neurons.append(current_layer_neurons)
        
        print(f"    Column {self.name} initialized with {neuron_name_counter} neurons in {len(self.neurons)} layers.")
    
    def _get_vertex_name(self, vertex: Any) -> int:
        if vertex not in self.global_vertex_map:
            v_id = self.next_vertex_id
            self.global_vertex_map[vertex] = v_id
            self.next_vertex_id += 1
            return v_id
        
        return self.global_vertex_map.get(vertex)
    
    def _process_data(self, data: list[Any], modulation_signal: float) -> Dict[str, Dict[int, Any]]:
        """
        Performs an ascending pass through the column, activating neurons.

        Collects local energy F and maximum stability P_single
        from the neurons of the **bottom** layer for each input element.

        Args:
            data: List of input elements (numeric or other hashable types).
            modulation_signal: Modulation signal (0.0 for predict, 1.0 for train).

        Returns:
            A dictionary with results for each element of the original data:
            {
                'element_f_local': {element_idx: [F1, F2,...]}, # List of F from pairs with this element
                'element_p_single': {element_idx: max_P_single} # Max. P_single for this element
            }
            The dictionaries may be incomplete if some elements were not processed.
        """
        num_original_elements = len(data)
        element_results = {
            'element_f_local': defaultdict(list),
            'element_p_single': defaultdict(float)  # default 0.0
        }
        
        if num_original_elements <= 1:
            return dict(element_results)  # Convert to dict before returning
        initial_bigrams = get_bigrams_zip(data)
        if not initial_bigrams:
            return dict(element_results)
        
        num_layers = len(self.neurons)
        if num_layers == 0:  # In case neurons were not created
            print(f"Warning Col {self.name}: No neuron layers found in ru_column.")
            return dict(element_results)
        bottom_layer_idx = num_layers - 1
        
        current_processing_input: List[Dict] = [
            {'pair': b, 'involved_indices': {i, i + 1}} for i, b in enumerate(initial_bigrams)
        ]
        
        # Iterate through layers from bottom to top (by indices in self.neurons)
        for layer_idx in range(num_layers - 1, -1, -1):  # From N-2 to 0
            if layer_idx >= len(self.neurons):
                continue  # Protection against out-of-bounds access
            layer_neurons = self.neurons[layer_idx]
            is_bottom_layer = (layer_idx == bottom_layer_idx)
            
            layer_outputs_for_next_layer = []
            # Ensure we do not try to process more pairs than there are neurons OR inputs
            num_pairs_to_process = min(len(current_processing_input), len(layer_neurons))
            
            for ni in range(num_pairs_to_process):
                input_info = current_processing_input[ni]
                neuron = layer_neurons[ni]
                
                x1, x2, z_output = None, None, -1
                f_value = float('inf');
                p1_res, p2_res = 0.0, 0.0
                involved_indices = set();
                original_idx1, original_idx2 = -1, -1
                
                try:
                    if is_bottom_layer:
                        input_dict = input_info  # This should be a dictionary
                        x1, x2 = input_dict['pair']
                        involved_indices = input_dict['involved_indices']
                        original_idx1, original_idx2 = ni, ni + 1
                    else:  # Upper layers
                        input_tuple = input_info  # This should be a tuple of two dictionaries
                        x1 = input_tuple[0]['z']
                        x2 = input_tuple[1]['z']
                        involved_indices = input_tuple[0]['involved_indices'] | input_tuple[1]['involved_indices']
                    
                    x1, x2 = self._get_vertex_name(x1), self._get_vertex_name(x2)
                    # Neuron activation
                    activation_result = neuron.ascending_activation(
                        x1, x2, modulation_signal=modulation_signal
                    )
                    
                    # Process the result (can be None in case of an internal error)
                    if activation_result is not None:
                        S, P_pair, F, b0, z_out, p1_from_n, p2_from_n = activation_result
                        f_value = F if F is not None else float('inf')
                        z_output = z_out if z_out is not None else -1
                        # Check p1/p2 for None before use
                        p1_res = p1_from_n if p1_from_n is not None else 0.0
                        p2_res = p2_from_n if p2_from_n is not None else 0.0
                    else:
                        # If ascending_activation returned None - there was an error
                        print(f"Warning Col {self.name} L{layer_idx} N{ni}: ascending_activation returned None.")
                        # Keep default values (f=inf, z=-1, p=0)
                
                except (KeyError, IndexError, TypeError) as e_input:
                    print(
                        f"Warning Col {self.name} L{layer_idx} N{ni}: Error accessing input data: {e_input}. Skipping.")
                    continue  # Skip this neuron/pair
                except Exception as e_act:
                    print(f"ERROR Col {self.name} L{layer_idx} N{ni}: Neuron activation failed: {e_act}")
                    traceback.print_exc()
                    # Set default/error values
                    f_value = float('inf');
                    z_output = -1;
                    p1_res = 0.0;
                    p2_res = 0.0
                    # Try to restore involved_indices if possible
                    if isinstance(input_info, dict):
                        involved_indices = input_info.get('involved_indices', set())
                    elif isinstance(input_info, tuple) and len(input_info) > 1:
                        involved_indices = input_info[0].get('involved_indices', set()) | input_info[1].get(
                            'involved_indices', set())
                
                # Collect results from the bottom layer
                if is_bottom_layer:
                    f_valid = isinstance(f_value, (int, float)) and not math.isinf(f_value) and not math.isnan(
                        f_value) and f_value >= 0
                    if f_valid:
                        # Add F for both original indices
                        if 0 <= original_idx1 < num_original_elements:
                            element_results['element_f_local'][original_idx1].append(f_value)
                        if 0 <= original_idx2 < num_original_elements:
                            element_results['element_f_local'][original_idx2].append(f_value)
                    
                    # Add P_single (already checked for None)
                    if 0 <= original_idx1 < num_original_elements:
                        element_results['element_p_single'][original_idx1] = max(
                            element_results['element_p_single'][original_idx1], p1_res)
                    if 0 <= original_idx2 < num_original_elements:
                        element_results['element_p_single'][original_idx2] = max(
                            element_results['element_p_single'][original_idx2], p2_res)
                
                # Prepare output for the next layer
                if z_output != -1:
                    # Ensure involved_indices is a set
                    if not isinstance(involved_indices, set):
                        involved_indices = set()
                    layer_outputs_for_next_layer.append({'z': z_output, 'involved_indices': involved_indices})
            
            # Prepare input for the next layer
            if len(layer_outputs_for_next_layer) <= 1:
                break
            current_processing_input = get_bigrams_zip(layer_outputs_for_next_layer)
        
        return {
            'element_f_local': dict(element_results['element_f_local']),
            'element_p_single': dict(element_results['element_p_single'])
        }
    
    def predict(self, data: list[Any]) -> List[Dict[str, float]]:
        """
        Calculates detailed familiarity scores for each element of the input data.

        Does not change the internal state of the column (activates neurons with modulation_signal=0.0).

        Args:
            data: List of input elements.

        Returns:
            A list of dictionaries, one for each input element, with the following keys:
            - 'p_single': Maximum P_single stability for this element.
            - 'min_F_local': Minimum free energy F, observed
                             in pairs involving this element on the bottom layer.
            - 'final_fs': Final familiarity score of the element [0, 1],
                          calculated as a probabilistic OR of familiarity from
                          the local context (based on min_F_local) and
                          the stability of the element itself (p_single).
            Returns -1.0 or inf for values in case of calculation errors for an element.
        """
        num_original_elements = len(data)
        default_output = {'p_single': 0.0, 'min_F_local': float('inf'), 'final_fs': 0.0}
        if num_original_elements < 1:
            return [default_output.copy() for _ in range(num_original_elements)]
        
        # Get current F and P_single without training
        element_results = self._process_data(data, modulation_signal=0.0)
        
        detailed_element_info: List[Dict[str, float]] = []
        for i in range(num_original_elements):
            f_locals_i = element_results['element_f_local'].get(i, [])
            # The p_single value is already the maximum, take it or 0.0
            p_single_i = element_results['element_p_single'].get(i, 0.0)
            
            # P_single clipped
            p_single_clipped = max(0.0, min(1.0, p_single_i))
            
            # Min_F_local
            valid_f_local = [f for f in f_locals_i if
                             isinstance(f, (int, float)) and not math.isinf(f) and not math.isnan(f) and f >= 0]
            min_F_local = min(valid_f_local) if valid_f_local else float('inf')
            
            # Final_FS
            fs_from_local_context = 0.0
            k_f = K_FAMILIARITY  # Use the global constant or self.k_familiarity if it exists
            if min_F_local != float('inf'):
                fs_calculated = calculate_familiarity_score(min_F_local, k=k_f)
                fs_from_local_context = fs_calculated  # Assign
            
            calculated_value = fs_from_local_context + p_single_clipped - (fs_from_local_context * p_single_clipped)
            final_fs_element = max(0.0, min(1.0, calculated_value))
            
            detailed_element_info.append({
                'p_single': p_single_clipped,
                'min_F_local': min_F_local,
                'final_fs': final_fs_element
            })
        
        # print("--- FINAL PREDICT OUTPUT ---")
        # print(f'DEBUG Column "{self.name}" says {detailed_element_info}')
        
        return detailed_element_info
    
    def predict_element_until_found(self, element: Any) -> float:
        try:
            bottom_neurons_layer = self.neurons[-1]
            for neuron in bottom_neurons_layer:
                response = neuron.single_stability(element)
                if response > 0.:
                    return response
        except IndexError:
            ...
        
        return 0.
    
    @staticmethod
    def _calculate_overall_fs(
            column_response: Optional[List[Optional[Dict[str, float]]]]
    ) -> float:
        """
        Calculates the average 'final_fs' (Overall FS) for the column's response to a window.
        This is a helper method for Column.train.

        Args:
            column_response: A list of dictionaries (or None) with the predict results
                               from this column for each element of the window.

        Returns:
            The average 'final_fs' value across all valid elements in the window,
            or 0.0 if the response is incorrect or does not contain valid scores.
        """
        overall_fs = 0.0
        
        # Check for None or an empty list
        if not column_response:
            # neuron_logger.warning(f"Col {self.name}: Cannot calculate Overall FS from None or empty response.") # Removed log for clarity
            return 0.0
        
        valid_final_fs_scores = []
        for element_data in column_response:
            # Check that the element is a dictionary and contains a valid final_fs
            if isinstance(element_data, dict):
                fs = element_data.get('final_fs')
                if isinstance(fs, (int, float)) and math.isfinite(fs):
                    # Add only valid scores (including 0.0)
                    valid_final_fs_scores.append(fs)
            # Ignore None or incorrect entries within the list
        
        if valid_final_fs_scores:  # If there is at least one valid value
            try:
                overall_fs = sum(valid_final_fs_scores) / len(valid_final_fs_scores)
            except ZeroDivisionError:  # In case valid_final_fs_scores became empty (unlikely)
                overall_fs = 0.0
        # else: If there are no valid scores, overall_fs remains 0.0
        
        # Return float, convert just in case (though division usually yields float)
        return float(overall_fs)
    
    # === MODIFIED TRAIN ===
    def train(self, data: list[Any], modulation_signal: float = 1.0) -> None:
        """
        Updates the column's state based on input data, considering modulation.
        Updates Overall_FS history and step counter if modulation is significant.
        """
        if len(data) <= 1:
            return  # Nothing to train on empty or single data items
        
        # 1. Call internal pass with the provided modulation_signal
        try:
            # The _process_data call itself activates training in neurons with the necessary modulation
            element_results = self._process_data(data, modulation_signal=modulation_signal)
            # The element_results are not needed here; they are only for predict
        except Exception as e_proc:
            print(f"ERROR Col {self.name}: _process_data failed during train: {e_proc}")
            traceback.print_exc()
            return  # Interrupt training on error
        
        # 2. Update history and counter ONLY IF there was actual training
        # Use the neuron's min_modulation_for_history hyperparameter
        # Neuron hyperparameters need to be passed to the column or be accessible
        # For now, use a fixed value of 0.1 (as in SimplicialNeuron default)
        min_mod_hist = getattr(self.hyperparameters, 'min_modulation_for_history', 0.1) if hasattr(self,
                                                                                                   'hyperparameters') else 0.1
        
        if modulation_signal >= min_mod_hist:
            self.training_steps_count += 1  # Increment the COLUMN's training step counter
            try:
                # Get predictions AFTER training to calculate Overall_FS
                predictions = self.predict(data)  # Call predict without modulation
                current_overall_fs = self._calculate_overall_fs(predictions)
                if math.isfinite(current_overall_fs):
                    self.overall_fs_history.append(current_overall_fs)
                    # print(f"Col {self.name}: Trained (m={modulation_signal:.3f}). Step: {self.training_steps_count}, Added OverallFS: {current_overall_fs:.4f}") # Debug
            except Exception as e_post_train:
                print(f"ERROR Col {self.name}: Failed to get/update Overall_FS after training: {e_post_train}")
                traceback.print_exc()
        # else: If modulation was low, do not update history
    
    # def train(self, data: list[Any], modulation_signal: float = 1.0) -> None:
    #     """
    #     Updates the internal state of the column (neurons) based on the input data.
    #
    #     Calls an internal pass with `modulation_signal=1.0` to activate training.
    #     Does not return metric values.
    #
    #     Args:
    #         data: List of input elements for training.
    #     """
    #     if len(data) <= 1:
    #         return
    #     # Simply call _process_data for training, the result is not important
    #     self._process_data(data, modulation_signal=modulation_signal)
    #
    #     # 2. Get current predictions AFTER training
    #     try:
    #         # Call predict to get final final_fs scores
    #         predictions = self.predict(data)  # predict calls _process_data with modulation=0.0
    #
    #         # 3. Calculate Overall_FS
    #         valid_fs = [p.get('final_fs', 0.0) for p in predictions if
    #                     p.get('final_fs', -1.0) >= 0.0]  # Consider valid fs
    #         current_overall_fs = sum(valid_fs) / len(valid_fs) if valid_fs else 0.0
    #
    #         # 4. Update history and counter
    #         self.overall_fs_history.append(current_overall_fs)
    #         self.training_steps_count += 1
    #         # column_manager_logger.debug(f"Col {self.name}: Trained. Step: {self.training_steps_count}, Added OverallFS: {current_overall_fs:.4f}")
    #
    #     except Exception as e_post_train:
    #         # If prediction after training failed, do not update history
    #         print(f"ERROR Col {self.name}: Failed to get/update Overall_FS after training: {e_post_train}")
    #         traceback.print_exc()
    
    def dump(self):
        # initial_meta = {
        #     'input_width': input_width,
        #     'max_overall_fs_history_len': 30,
        #     'overall_fs_history': [],
        #     'training_steps_count': 0,
        #     'global_vertex_map': {},
        #     'next_vertex_id': 0
        # }
        """Saves the state of all neurons in the column and its metadata."""
        print(f"        Column {self.name}: Dumping state...")
        save_errors = 0
        for layer in self.neurons:
            for neuron in layer:
                try:
                    neuron.dump()
                except Exception as e:
                    print(f"        Error dumping N {neuron.name} in Col {self.name}: {e}")
                    save_errors += 1
        # Save metadata
        if self.meta_storage:
            try:
                # Update data before saving
                self.meta_storage.data['input_width'] = self.input_width
                self.meta_storage.data['max_overall_fs_history_len'] = self.max_overall_fs_history_len
                self.meta_storage.data['overall_fs_history'] = self.overall_fs_history
                self.meta_storage.data['training_steps_count'] = self.training_steps_count
                self.meta_storage.data['global_vertex_map'] = self.global_vertex_map
                self.meta_storage.data['next_vertex_id'] = self.next_vertex_id
                self.meta_storage.dump()
            except Exception as e:
                print(f"        Warning: Col {self.name} meta save error: {e}")
        else:
            print(f"        Warning: Col {self.name} cannot save metadata (storage object missing).")
        
        status = "successfully" if save_errors == 0 else f"with {save_errors} errors"
        print(f"        Column {self.name}: State dump finished {status}.")


# --- End of Column class ---


# --- Test for Column ---
if __name__ == '__main__':
    # --- Test Parameters ---
    TEST_STORAGE_PATH_RU = Path('./data/col0')
    TEST_STORAGE_PATH_EN = Path('./data/col1')
    COL_NAME = 0
    INPUT_WIDTH = 24  # Window width = column width
    WINDOW_STEP = 4
    USE_ORD = True  # We will use ord() for neuron inputs
    NUM_TRAIN_WINDOWS = 100  # Number of windows for training
    WINDOW_TO_PREDICT = 0  # Index of the window for which prediction will be made
    
    with open(Path('../assets/lang/ru/ru_dict_prepared.txt'), 'r', encoding='utf-8') as file:
        TEXT_DATA_RU = file.read()
    
    with open(Path('../assets/lang/en/en_dict_prepared.txt'), 'r', encoding='utf-8') as file:
        TEXT_DATA_EN = file.read()
    
    # --- Cleanup and Column Creation ---
    if TEST_STORAGE_PATH_RU.exists():
        print(f"Cleaning up test data in {TEST_STORAGE_PATH_RU}...")
        shutil.rmtree(TEST_STORAGE_PATH_RU)
    
    if TEST_STORAGE_PATH_EN.exists():
        print(f"Cleaning up test data in {TEST_STORAGE_PATH_EN}...")
        shutil.rmtree(TEST_STORAGE_PATH_EN)
    
    print(f"Initializing Column {COL_NAME} with input_width={INPUT_WIDTH}...")
    try:
        # Pass the PATH TO THE FOLDER WHERE THIS COLUMN'S FOLDER WILL BE LOCATED (storage_path / f'col_{name}')
        # But Column expects storage_path - the directory WHERE column folders will be
        # Therefore, create a dummy parent directory
        parent_storage_path_ru = TEST_STORAGE_PATH_RU.parent / f"{TEST_STORAGE_PATH_RU.name}_parent"
        parent_storage_path_en = TEST_STORAGE_PATH_EN.parent / f"{TEST_STORAGE_PATH_EN.name}_parent"
        ru_column = Column(name=0, input_width=INPUT_WIDTH, storage_path=parent_storage_path_ru)
        en_column = Column(name=1, input_width=INPUT_WIDTH, storage_path=parent_storage_path_en)
        print("Column initialized.")
    except Exception as e:
        print(f"Failed to initialize Column: {e}")
        exit()
    
    # --- Training on Sliding Window ---
    print(f"\n--- Training on {NUM_TRAIN_WINDOWS} windows ---")
    training_windows_ru = []
    training_windows_en = []
    window_generator_ru = sliding_window.sliding_window(TEXT_DATA_RU, INPUT_WIDTH, WINDOW_STEP)
    window_generator_en = sliding_window.sliding_window(TEXT_DATA_EN, INPUT_WIDTH, WINDOW_STEP)
    
    # Train the Cyrillic column
    for i, window_tuple in enumerate(window_generator_ru):
        if i >= NUM_TRAIN_WINDOWS:
            break
        window_str = "".join(window_tuple)
        training_windows_ru.append(window_tuple)  # Save for prediction
        print(f"Training window {i + 1}/{NUM_TRAIN_WINDOWS}: '{window_str}'")
        
        # Prepare data for the column
        if USE_ORD:
            input_data = [c for c in window_tuple]
        else:
            input_data = list(window_tuple)
        
        if len(input_data) < 2:  # Skip windows that are too short (due to remainder)
            print("  Window too short, skipping.")
            continue
        
        try:
            ru_column.train(input_data)
        except Exception as e:
            print(f"  Error training on window: {e}")
            # Continue with the next window
    
    # Train the Latin column
    for i, window_tuple in enumerate(window_generator_en):
        if i >= NUM_TRAIN_WINDOWS:
            break
        window_str = "".join(window_tuple)
        training_windows_en.append(window_tuple)  # Save for prediction
        print(f"Training window {i + 1}/{NUM_TRAIN_WINDOWS}: '{window_str}'")
        
        # Prepare data for the column
        if USE_ORD:
            input_data = [c for c in window_tuple]
        else:
            input_data = list(window_tuple)
        
        if len(input_data) < 2:  # Skip windows that are too short (due to remainder)
            print("  Window too short, skipping.")
            continue
        
        try:
            en_column.train(input_data)
        except Exception as e:
            print(f"  Error training on window: {e}")
            # Continue with the next window
    
    # --- Prediction for One Cyrillic Window ---
    if len(training_windows_ru) > WINDOW_TO_PREDICT:
        print(f"\n--- Predicting for window {WINDOW_TO_PREDICT + 1} ---")
        predict_window_tuple = training_windows_ru[WINDOW_TO_PREDICT]
        predict_window_str = "".join(predict_window_tuple)
        print(f"Input window: '{predict_window_str}'")
        
        if USE_ORD:
            predict_input_data = [c for c in predict_window_tuple]
        else:
            predict_input_data = list(predict_window_tuple)
        
        if len(predict_input_data) >= 2:
            try:
                print("\nDetailed Predictions (per element):")
                try:
                    detailed_predictions_from_ru_column = ru_column.predict(predict_input_data)
                    # print("\nDetailed Predictions (per element):")
                    # --- ADD LENGTH CHECK ---
                    # print(
                    # f"DEBUG Main: len(detailed_predictions_from_ru_column)={len(detailed_predictions_from_ru_column)}, len(predict_window_tuple)={len(predict_window_tuple)}")
                    # --- END OF LENGTH CHECK ---
                    
                    # --- ADD DEBUGGING OF detailed_predictions_from_ru_column CONTENT ---
                    # print(f"DEBUG Main: Raw detailed_predictions_from_ru_column list:\n{detailed_predictions_from_ru_column}\n")
                    # --- END OF DEBUGGING ---
                    
                    if len(detailed_predictions_from_ru_column) == len(predict_window_tuple):
                        for i, char in enumerate(predict_window_tuple):
                            preds = detailed_predictions_from_ru_column[i]
                            print(f"  '{char}' (idx {i}): "
                                  f"p_single={preds.get('p_single', -1):.3f}, "
                                  f"min_F_local={preds.get('min_F_local', float('inf')):.3f}, "  # <--- FIXED
                                  f"final_fs={preds.get('final_fs', -1):.3f}")
                    else:
                        print("  Error: Prediction list length mismatch.")
                except Exception as e:
                    print(f"  Error predicting window: {e}")
            except Exception as e:
                print(f"  Error predicting window: {e}")
        else:
            print("  Window too short for prediction.")
    
    # --- Prediction for One Latin Window ---
    if len(training_windows_en) > WINDOW_TO_PREDICT:
        print(f"\n--- Predicting for window {WINDOW_TO_PREDICT + 1} ---")
        predict_window_tuple = training_windows_en[WINDOW_TO_PREDICT]
        predict_window_str = "".join(predict_window_tuple)
        print(f"Input window: '{predict_window_str}'")
        
        if USE_ORD:
            predict_input_data = [c for c in predict_window_tuple]
        else:
            predict_input_data = list(predict_window_tuple)
        
        if len(predict_input_data) >= 2:
            try:
                print("\nDetailed Predictions (per element):")
                try:
                    detailed_predictions_from_ru_column = en_column.predict(predict_input_data)
                    # print("\nDetailed Predictions (per element):")
                    # --- ADD LENGTH CHECK ---
                    # print(
                    # f"DEBUG Main: len(detailed_predictions_from_ru_column)={len(detailed_predictions_from_ru_column)}, len(predict_window_tuple)={len(predict_window_tuple)}")
                    # --- END OF LENGTH CHECK ---
                    
                    # --- ADD DEBUGGING OF detailed_predictions_from_ru_column CONTENT ---
                    # print(f"DEBUG Main: Raw detailed_predictions_from_ru_column list:\n{detailed_predictions_from_ru_column}\n")
                    # --- END OF DEBUGGING ---
                    
                    if len(detailed_predictions_from_ru_column) == len(predict_window_tuple):
                        for i, char in enumerate(predict_window_tuple):
                            preds = detailed_predictions_from_ru_column[i]
                            print(f"  '{char}' (idx {i}): "
                                  f"p_single={preds.get('p_single', -1):.3f}, "
                                  f"min_F_local={preds.get('min_F_local', float('inf')):.3f}, "  # <--- FIXED
                                  f"final_fs={preds.get('final_fs', -1):.3f}")
                    else:
                        print("  Error: Prediction list length mismatch.")
                except Exception as e:
                    print(f"  Error predicting window: {e}")
            except Exception as e:
                print(f"  Error predicting window: {e}")
        else:
            print("  Window too short for prediction.")
    
    # # --- Saving ---
    # print("\n--- Saving Column State ---")
    # try:
    #     ru_column.dump()
    # except Exception as e:
    #     print(f"  Error dumping ru_column state: {e}")
    
    print("\n--- Test Finished ---")
    
    # test_window = ['а', ' ', 'б']
    # detailed_predictions_from_ru_column = ru_column.predict([c for c in test_window])
    # for i, dp in enumerate(detailed_predictions_from_ru_column):
    #     sym = test_window[i]
    #     print(f'{sym}: {dp}')
    
    mixed_window = ['а', 'б', 'в', 'г', 'д', ' ', 'з', 'ж', 'i', 'j', 'k', 'l', ' ', 'n', 'к', 'л', 'м', 'н', 'о', 'з',
                    'u', 'v', 'w', 'x']
    detailed_predictions_from_ru_column = ru_column.predict([c for c in mixed_window])
    for i, dp in enumerate(detailed_predictions_from_ru_column):
        sym = mixed_window[i]
        print(f'{sym}: {dp}')
    
    detailed_predictions_from_en_column = en_column.predict([c for c in mixed_window])
    for i, dp in enumerate(detailed_predictions_from_en_column):
        sym = mixed_window[i]
        print(f'{sym}: {dp}')