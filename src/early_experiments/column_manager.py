import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import traceback

from collections import defaultdict, deque
from pathlib import Path
from typing import (
    Optional,
    Dict,
    Tuple,
    Deque,
    List,
    Any,
    Set
)
from dataclasses import dataclass

from stabiliser import Stabilizer
from binstorage import BinaryStorage
from column import Column
from namer import Namer


@dataclass
class ColumnManagerHyperParams:
    """
    Hyperparameters for configuring the ColumnManager's operation.
    """
    
    # --- Main Thresholds and Decision Parameters ---
    vigilance: float = 0.35
    # Base vigilance threshold. Used for:
    # 1. Determining if a STABLE winner handled the window (for Global Mismatch).
    # 2. Determining if a column has reached an acceptable performance level to
    #    be considered stable (in the _is_column_stable_trend / _get_stability_metrics method).
    # 3. As a base threshold in other comparison logics (e.g., adaptive masking).
    
    # global_mismatch_factor: float = 0.5 # DEPRECATED? Appears to be replaced by vigilance check.
    
    # --- History Parameters ---
    prediction_history_size: int = 1000
    # Maximum length of the prediction history (`Deque`), stored in the
    # `ColumnManager.prediction_history` attribute for each column.
    # The specific use of this history depends on the further implementation
    # of the `process` method. Possibly for analysing prediction dynamics or other purposes.
    
    column_fs_history_len: int = 30
    # Maximum length of the Overall_FS history stored WITHIN each column (`Column.overall_fs_history`).
    # Used for calculating column stability based on trend.
    
    # --- Stability Parameters ---
    stability_trend_window_k: int = 15
    # Window size (number of recent training steps/Overall_FS values from `Column.overall_fs_history`),
    # used for calculating the column's stability trend. Must be >= 2.
    
    stability_slope_threshold: float = 0.005
    # Maximum absolute slope of the Overall_FS trend, at which a column
    # can be considered stable (if its average FS is also above vigilance).
    
    # stability_window_k: int = 15       # DEPRECATED. Old plateau finding method.
    # stability_threshold_factor: float = 1. # DEPRECATED. Old plateau finding method.
    # min_plateau_len: int = 5           # DEPRECATED. Old plateau finding method.
    
    # --- Masked Training and Context Parameters ---
    learning_context_k: int = 1
    # Context radius (k neighbours to the left and right), which is included
    # in the mask for INITIAL training of a NEW column (mask_new).
    
    train_existing_context_k: Optional[int] = 0  # DEPRECATED? Did we decide not to train old columns with a mask?
    # Context radius for the mask for RETRAINING an EXISTING (old) column (mask_old)
    # at the moment a new one is created. Possibly not used in the current plan.
    
    # --- Local Column Creation and Retraining Parameters ---
    min_segment_len_new: int = 3
    # Minimum length of a continuous segment of "poorly recognised"
    # elements (`poor_indices`), required for creating a new
    # LOCAL column in Step 6.
    
    min_segment_len_train: int = 2
    # Minimum length of a continuous segment assigned to a column,
    # for that column to be RETRAINED on this segment in Step 7.
    
    # --- Incubation Parameters (Will be added/used later) ---
    # incubator_base_similarity_threshold: float = 0.3
    # incubator_min_similarity_threshold: float = 0.05
    # incubator_max_unstable_slope: float = 0.1
    # incubator_max_steps: int = 50
    
    # --- Adaptive Masking Parameters (for _create_training_masks) ---
    std_dev_factor_for_mask: float = 1.5
    # Coefficient (N sigmas) used for calculating the adaptive masking threshold
    # (threshold = mean_fs - N * std_dev_fs).
    
    min_fs_for_mask: float = 0.1
    # Minimum absolute value for the adaptive masking threshold.
    
    # --- ID Set Parameters (for Step 8, will be added/used later) ---
    # common_element_threshold: float = 0.6
    # common_set_min_columns: int = 2
    
    # --- Dissonance Parameters ---
    dissonance_history_len: int = 50  # History length for baseline entropy
    dissonance_factor: float = 2.5  # Factor for comparing current entropy with baseline
    min_idealness_threshold: float = 0.2  # Threshold for idealness of unstable columns
    
    # --- Unstable Protection Parameter ---
    # Renamed for clarity regarding Overall_FS
    min_overall_fs_for_unstable_protection: float = 0.05
    
    softmax_temp: float = 1.0  # Temperature for entropy calculation
    
    common_set_min_columns: int = 2
    
    relative_confidence_threshold: float = 0.1  # Softmax probability threshold for an element
    softmax_temp_relative: float = 0.1  # Softmax temperature over the window for the column
    
    threshold_novelty: float = 0.7  # SumIdealness threshold for intensive learning mode
    min_learning_rate: float = 0.02  # Base learning rate
    softmax_temp_modulation: float = 0.05  # Temperature for learning modulation
    
    novelty_history_len: int = 50  # History length for baseline idealness
    novelty_detection_factor: float = 0.8  # Factor for comparing current idealness with baseline (has it dropped below 80%?)
    common_element_threshold: float = 0.6  # For ID sets
    
    top_down_boost_factor: float = 0.25


class ColumnManagerEmergent:
    def __init__(
            self,
            name: int,
            storage_path: Path,
            input_width=24,
            columns_quantity=10,
            hyperparameters: ColumnManagerHyperParams = ColumnManagerHyperParams()
    ):
        c_metadata = {
            'column_ids': [],
            'dissonance_history': deque(maxlen=hyperparameters.dissonance_history_len),
            'last_activity_steps': {},
            'max_idealness_history': [],
            'current_step': 0,
            'novelty_history': []
        }
        
        c_manager_folder = f'c_manager_{name}'
        self.full_path = storage_path / c_manager_folder / f'c_manager_{name}_meta.bin'
        self.columns_path = storage_path / c_manager_folder / 'columns'
        self.namers_path = storage_path / c_manager_folder / 'namers'
        self.columns_path.mkdir(parents=True, exist_ok=True)
        self.metadata = BinaryStorage(self.full_path, c_metadata)
        self.hyper_params = hyperparameters
        self.last_activity_steps = self.metadata.data.get('last_activity_steps', {})
        self.stabilizer = Stabilizer(0, storage_path / c_manager_folder, columns_quantity)
        
        # Initialisation of namers
        input_namer_name = 0
        column_namer_name = 1
        self.input_namer = Namer(input_namer_name, self.namers_path)
        self.column_namer = Namer(column_namer_name, self.namers_path)
        
        # Important: reset prediction history when loading state
        self.prediction_history: Dict[int, Deque[Tuple[int, List[float]]]] = defaultdict(
            lambda: deque(maxlen=self.hyper_params.prediction_history_size)
        )
        
        self.current_step = self.metadata.data.get('current_step', 0)
        self.last_activity_steps: Dict[int, int] = {}
        self.input_width = input_width
        
        self.dissonance_history: Deque[float] = self.metadata.data.get('dissonance_history', deque(
            maxlen=self.hyper_params.dissonance_history_len))
        
        self.column_ids = self.metadata.data.get('column_ids', c_metadata['column_ids'])
        self.columns: Dict[int, Column] = {}
        
        self.history_sum_idealness: Dict[int, List[Optional[float]]] = defaultdict(
            list)  # {col_id: [sum_id_step1, sum_id_step2, ...]}
        self.history_modulation: Dict[int, List[Optional[float]]] = defaultdict(
            list)  # {col_id: [mod_step1, mod_step2, ...]}
        self.history_steps: List[int] = []  # List of step numbers
        
        self.name = name
        
        raw_novelty_history = self.metadata.data.get('novelty_history', [])
        
        # History of MAXIMUM idealness
        raw_max_idealness_history = self.metadata.data.get('max_idealness_history', [])
        if isinstance(raw_max_idealness_history, list):
            self.max_idealness_history: Deque[float] = deque(raw_max_idealness_history,
                                                             maxlen=self.hyper_params.novelty_history_len)  # Use the same length parameter
        else:
            self.max_idealness_history = deque(maxlen=self.hyper_params.novelty_history_len)
        
        # Type check before creating deque
        if isinstance(raw_novelty_history, list):
            self.novelty_history: Deque[float] = deque(raw_novelty_history,
                                                       maxlen=self.hyper_params.dissonance_history_len)  # Renamed
        else:  # Handling old format or error
            print("WARN: Could not load novelty_history, initializing empty deque.")
            self.novelty_history: Deque[float] = deque(maxlen=self.hyper_params.dissonance_history_len)
        
        print(f"ColumnManager: {len(self.column_ids)} column(s) exist")
        
        if len(self.column_ids) > 0:
            # Loading columns
            loaded_count = 0
            for column_id in self.column_ids:
                print(f"  Attempting to load Column {column_id}...")
                try:
                    column = Column(name=column_id, input_width=self.input_width, storage_path=self.columns_path)
                    self.columns[column_id] = column
                    self.prediction_history[column_id] = deque(maxlen=self.hyper_params.prediction_history_size)
                    loaded_count += 1
                    print(f"    Successfully loaded Column {column_id}")
                
                except Exception as e:
                    print(f"    Error loading Col {column_id}: {e}. Skipping.")
                    traceback.print_exc()
            
            print(f"ColumnManager: Loaded {loaded_count} column(s).")
        
        else:
            # No columns exist, creating initial pool...
            print(f'No columns exist, creating initial pool...')
            for c_name in range(0, columns_quantity):
                new_column = Column(c_name, self.input_width, self.columns_path)
                self.columns[c_name] = new_column
                self.column_ids.append(c_name)
                self.column_ids.sort()
    
    @staticmethod
    def _calculate_idealness_score(
            element_response: Optional[Dict[str, float]]
    ) -> float:
        """
        Calculates the "idealness" score of the response for a single element.
        Uses the PRODUCT of the element's familiarity (p_single)
        and context familiarity (1/(1+mfl)).
        Ideal = 1.0 (only if p_single=1.0 AND min_F_local=0.0).
        Result = 0.0 if EITHER p_single=0, OR mfl=inf (or incorrect).

        Args:
            element_response: Dictionary with neuron predict results for the element
                             (e.g., {'p_single': 0.8, 'min_F_local': 0.1, 'final_fs': 0.9})
                             or None.

        Returns:
            Idealness score in the range [0.0, 1.0].
        """
        # Check for empty or incorrect input
        if not isinstance(element_response, dict):
            return 0.0
        
        p_single = element_response.get('p_single')
        min_f_local = element_response.get('min_F_local')
        
        # --- Validation and retrieval of p_single ---
        # If values are incorrect, p_single is considered zero
        if not (isinstance(p_single, (int, float)) and math.isfinite(p_single) and 0.0 <= p_single <= 1.0):
            valid_p_single = 0.0
        else:
            valid_p_single = float(p_single)  # Ensure float
        
        # --- Calculation of context familiarity ---
        # If min_F_local is incorrect or infinite, context familiarity is zero
        if not (isinstance(min_f_local, (int, float)) and math.isfinite(min_f_local) and min_f_local >= 0.0):
            familiarity_from_local = 0.0
        else:
            # Using the function 1 / (1 + x)
            familiarity_from_local = 1.0 / (1.0 + float(min_f_local))  # Ensure float
        
        # === Calculation of Idealness via PRODUCT ===
        idealness = valid_p_single * familiarity_from_local
        
        # Final check and range limitation (although the product should already be within it)
        idealness = max(0.0, min(1.0, idealness))
        
        # --- Debug output (can be commented out) ---
        # print(f"    DEBUG idealness: p_single={valid_p_single:.3f}, mfl={min_f_local}, familiarity={familiarity_from_local:.3f} -> idealness={idealness:.3f}")
        # ------------------------------------------------
        
        return idealness
    
    def _update_max_idealness_history(self, current_max_idealness: float):
        if math.isfinite(current_max_idealness):
            self.max_idealness_history.append(current_max_idealness)
        else:
            print(f"  WARN: Attempted to add non-finite max_idealness {current_max_idealness} to history.")
    
    def _get_baseline_max_idealness(self) -> float:
        if not self.max_idealness_history:
            return 0.0  # Baseline 0, so that on the first step it is definitely > factor*baseline
        valid_history = [h for h in self.max_idealness_history if math.isfinite(h)]
        if not valid_history:
            return 0.0
        baseline: float = float(np.mean(valid_history))
        return baseline
    
    def _touch_intersection(self, reacted_columns: List[int]) -> int:
        if not reacted_columns:
            return -1
        sorted_columns_tuple = tuple(sorted(reacted_columns))
        intersection_id = self.column_namer.name_it(sorted_columns_tuple)
        return intersection_id
    
    @staticmethod
    def _calculate_overall_fs(
            column_response: Optional[List[Optional[Dict[str, float]]]]
    ) -> float:
        """
        Calculates the average 'final_fs' (Overall FS) for a column's response to a window.

        Args:
            column_response: List of dictionaries (or None) with predict results
                               from the column for each element of the window.

        Returns:
            Average 'final_fs' value across all valid elements in the window,
            or 0.0 if the response is incorrect or does not contain valid scores.
        """
        overall_fs = 0.0
        
        if not column_response:
            # If there is no response (None or empty list)
            return 0.0
        
        valid_final_fs_scores = []
        for element_data in column_response:
            if isinstance(element_data, dict):
                fs = element_data.get('final_fs')
                # Check that fs is a correct number (not NaN, not inf)
                if isinstance(fs, (int, float)) and math.isfinite(fs):
                    valid_final_fs_scores.append(fs)
            # Ignore None or incorrect entries within the list
        
        if valid_final_fs_scores:  # If there is at least one valid value
            overall_fs = sum(valid_final_fs_scores) / len(valid_final_fs_scores)
        # else: If there are no valid scores, overall_fs remains 0.0
        
        return overall_fs
    
    def _identify_common_elements_simplified(self, n_elements: int,
                                             final_preds: Dict[int, Optional[List[Optional[Dict[str, float]]]]]) -> \
            Dict[int, int]:
        """ Identifies common elements based on the final_fs of ALL columns > threshold. """
        common_elements_map: Dict[int, int] = {}
        recognition_threshold = self.hyper_params.common_element_threshold
        min_columns_for_common = self.hyper_params.common_set_min_columns
        
        for i in range(n_elements):
            confident_cols: Set[int] = set()
            for col_id, response_list in final_preds.items():  # Iterate over ALL columns
                if response_list and i < len(response_list) and isinstance(response_list[i], dict):
                    fs = response_list[i].get('final_fs')
                    if isinstance(fs, (int, float)) and math.isfinite(fs) and fs >= recognition_threshold:
                        confident_cols.add(col_id)  # Add the ID of the column that passed the threshold
            
            if len(confident_cols) >= min_columns_for_common:
                common_id = self._touch_intersection(list(confident_cols))
                if common_id != -1:
                    common_elements_map[i] = common_id
        # print(f"  DEBUG (_identify_common_elements_simplified): Found {len(common_elements_map)} common elements.")
        return common_elements_map
    
    def process(self, window: List[Any]) -> Optional[Tuple[int, List[Tuple[Any, Tuple[int, float]]]]]:
        original_chars = list(window)
        self.current_step += 1
        current_processing_step = self.current_step
        print(f"--- Processing Window (Step {current_processing_step}): {''.join(map(str, original_chars))} ---")
        
        # === Step 0: Obtaining Top-Down Prediction ===
        # The Stabilizer predicted this in the *previous* step t
        # and saved it for use in step t+1.
        # We should retrieve it here.
        predicted_distribution = self.stabilizer.predict_next_distribution()
        print(
            f"    DEBUG TopDown Prediction (for this step): { {k: f'{v:.3f}' for k, v in predicted_distribution.items() if v > 0.01} }")
        
        # === Step 1: Preparation ===
        # print("  Starting Step 1: Preprocessing...")
        try:
            cleaned_input: List[int] = [self.input_namer.name_it(v) for v in original_chars]
        except Exception as e:
            print(f"  ERROR: Input Naming failed: {e}. Skipping.")
            traceback.print_exc()
            return -1, [(c, (-1, 0.0)) for c in original_chars]
        n_elements = len(cleaned_input)
        if n_elements < 2:
            print(f"  Input too short ({n_elements} < 2).")
            return -1, [(original_chars[i], (-1, 0.0)) for i in range(n_elements)]
        if n_elements > self.input_width:
            original_chars = original_chars[:self.input_width]
            cleaned_input = cleaned_input[:self.input_width]
            n_elements = len(cleaned_input)
            print(f"  Input truncated to {n_elements} elements.")
        # print("  Finished Step 1.")
        
        # === Step 2: Response Evaluation and Success Assessment ===
        # print("  Starting Step 2: Response Evaluation...")
        all_preds: Dict[int, Optional[List[Optional[Dict[str, float]]]]] = {}
        all_sum_idealness: Dict[int, float] = {}
        max_idealness = -math.inf
        for col_id, col in self.columns.items():
            prediction = None
            current_sum = 0.0
            try:
                prediction = col.predict(cleaned_input)
                if prediction is not None and len(prediction) == n_elements:
                    all_preds[col_id] = prediction
                    if prediction:
                        for element_data in prediction:
                            idealness = self._calculate_idealness_score(element_data)  # PRODUCT!
                            if math.isfinite(idealness):
                                current_sum += idealness
                else:
                    all_preds[col_id] = None
            except Exception as e:
                print(f"    ERROR getting prediction from Col {col_id}: {e}")
                traceback.print_exc()
                all_preds[col_id] = None
            finally:
                all_sum_idealness[col_id] = current_sum
                # Find the maximum IMMEDIATELY to avoid iterating a second time
                if current_sum > max_idealness:
                    max_idealness = current_sum
        print(f"    Max Sum Idealness found: {max_idealness:.4f}")
        # print("  Finished Step 2.")
        
        # === Step 3: Modulation and Training All ===
        # print("  Starting Step 3: Modulation and Training...")
        modulation_signals: Dict[int, float] = {}  # Dictionary for storing final modulation
        # Flag for successful modulation calculation, can be removed if there is try-except
        # calculated_modulation = False
        is_novel_window = False  # Flag for history update
        
        # --- 3.1: Calculation of Novelty Degree ---
        # max_idealness was calculated at the end of Step 2
        baseline_max_idealness = self._get_baseline_max_idealness()
        # The threshold is now used only for calculating the novelty degree
        novelty_detection_threshold = baseline_max_idealness * self.hyper_params.novelty_detection_factor
        novelty_degree = 0.0  # [0, 1], where 1 is maximum novelty
        
        if baseline_max_idealness > 1e-9:  # Avoid division by zero
            gap = novelty_detection_threshold - max_idealness
            # Normalise the gap relative to the baseline level
            relative_gap = max(0.0, gap / baseline_max_idealness)
            # Novelty degree is the normalised gap, limited to [0, 1]
            novelty_degree = max(0.0, min(1.0, relative_gap))
        elif max_idealness < 1e-9:  # If both baseline is 0 and current is 0 - full novelty
            novelty_degree = 1.0
        # else: If baseline is 0 and current > 0, then novelty = 0 (as gap is negative)
        
        # Logging for debugging
        print(
            f"    Baseline Max Idealness: {baseline_max_idealness:.4f}, Factor: {self.hyper_params.novelty_detection_factor:.2f}, Novelty Threshold: {novelty_detection_threshold:.4f}")
        print(f"    Current Max Idealness: {max_idealness:.4f} -> Novelty Degree: {novelty_degree:.4f}")
        
        # Set the is_novel_window flag if novelty_degree is high (e.g., > 0.9?).
        # This is only needed to AVOID updating the max_idealness history later.
        # One can simply use the comparison of max_idealness with the threshold, as before.
        # Let's stick to comparing with the threshold for updating the history.
        is_novel_window = (max_idealness < novelty_detection_threshold)
        if is_novel_window:
            print(f"    Novelty detected (based on MaxIdealness vs Threshold). High plasticity mode likely.")
        
        # --- 3.2: Calculation of Base Modulation (Softmax by Idealness) ---
        # This calculation is still needed for interpolation
        base_modulation: Dict[int, float] = {}
        # Important: all_sum_idealness is a dictionary {col_id: sum_idealness}, calculated in Step 2
        # Get values in the order of column IDs
        col_ids_in_order = sorted(list(self.columns.keys()))  # Take current IDs and sort them
        all_idealness_values = np.array(
            [all_sum_idealness.get(k, -math.inf) for k in col_ids_in_order])  # Use sorted order
        
        min_lr = self.hyper_params.min_learning_rate
        max_lr = 1.0  # Maximum rate
        
        # Check if it makes sense to calculate Softmax at all
        if not np.any(np.isfinite(all_idealness_values)) or np.all(all_idealness_values <= -math.inf):
            print(
                "    WARN: All columns had non-finite/negative SumIdealness. Using min learning rate as base modulation.")
            for k_id in col_ids_in_order:
                base_modulation[k_id] = min_lr
        else:
            all_idealness_values[all_idealness_values == -math.inf] = -1e9  # Replacement for -inf
            temp = self.hyper_params.softmax_temp_modulation
            try:
                idealness_shifted = all_idealness_values - np.max(all_idealness_values)  # Stabilisation
                exp_idealness = np.exp(idealness_shifted / temp)
                sum_exp = np.sum(exp_idealness)
                # Calculation of probabilities (attention shares)
                attention_probs = exp_idealness / sum_exp if sum_exp > 1e-9 else np.zeros_like(all_idealness_values)
                learning_range = max_lr - min_lr
                # Calculation of base modulation
                for idx, k_id in enumerate(col_ids_in_order):
                    base_modulation[k_id] = min_lr + attention_probs[idx] * learning_range
                # print(f"      Base Modulation Signals (Softmax): { {k: f'{v:.3f}' for k, v in base_modulation.items()} }") # Debug
            except Exception as e:
                print(f"  ERROR calculating softmax for base modulation: {e}")
                for k_id in col_ids_in_order:
                    base_modulation[k_id] = min_lr  # Fallback to min_lr on error
        
        # --- 3.3: Calculation of Final Modulation (Interpolation + Top-Down) ---
        top_down_boost_factor = self.hyper_params.top_down_boost_factor
        for k_id in col_ids_in_order:
            expectedness_k = self.stabilizer.get_expectedness(predicted_distribution, k_id)
            current_base_mod = base_modulation.get(k_id, min_lr)
            # Apply Top-Down Bonus (multiplicatively)
            adjusted_base_mod = current_base_mod * (1.0 + top_down_boost_factor * expectedness_k)
            adjusted_base_mod = min(max_lr, adjusted_base_mod)  # Limit from above
            # Interpolate
            final_modulation = adjusted_base_mod * (1.0 - novelty_degree) + max_lr * novelty_degree
            modulation_signals[k_id] = max(min_lr, min(max_lr, final_modulation))
        
        # print(f"      Final Modulation Signals (sample): {dict(list(modulation_signals.items())[:3])}...") # Debug
        
        # --- 3.4: Training ---
        # print("    Training all columns with calculated modulation...")
        trained_cols_this_step = set()
        for k_id, modulation in modulation_signals.items():
            column_to_train = self.columns.get(k_id)
            # Train if modulation is ABOVE the minimum base (to avoid almost zero training)
            if column_to_train and modulation >= self.hyper_params.min_learning_rate * 1.01:  # Slight margin above min_lr
                try:
                    print(f"      Training Col {k_id} with modulation {modulation:.4f}")
                    # Pass the signal to train
                    column_to_train.train(cleaned_input, modulation_signal=modulation)
                    self.last_activity_steps[k_id] = current_processing_step  # Update activity
                    trained_cols_this_step.add(k_id)
                    # Overall FS history update now happens inside train
                except Exception as e:
                    print(f"    ERROR training Col {k_id}: {e}")
                    traceback.print_exc()
        # print(f"    Finished training columns: {trained_cols_this_step}")
        
        # print("  Finished Step 3: Modulation and Training.")
        # === End of Step 3 ===
        
        # === Step 4: Obtaining Final Responses ===
        # print("  Starting Step 4: Getting Final Predictions...") # Removed
        final_preds: Dict[int, Optional[List[Optional[Dict[str, float]]]]] = {}
        for col_id, col in self.columns.items():
            try:
                prediction = col.predict(cleaned_input)
                if prediction is not None and len(prediction) == n_elements:
                    final_preds[col_id] = prediction
                else:
                    final_preds[col_id] = None
            except Exception as e:
                print(f"  ERROR getting final prediction from Col {col_id}: {e}\") final_preds[col_id] = None")
            # print("  Finished Step 4.")
        
        # === Step 5: Final Segmentation, Output, and Stabilizer Update ===
        print("  Starting Step 5: Final Assignment, Stabilizer Update & Output...")
        # --- 5a: Element-wise Winners ---
        local_assignment: List[int] = [-1] * n_elements
        element_max_score: List[float] = [0.0] * n_elements
        for i in range(n_elements):
            best_k = -1
            max_fs = -math.inf
            for k in self.column_ids:  # Use the current list of IDs
                pred_list = final_preds.get(k)
                if pred_list and i < len(pred_list) and isinstance(pred_list[i], dict):
                    fs = pred_list[i].get('final_fs', -math.inf)
                    if isinstance(fs, (int, float)) and math.isfinite(fs):
                        if fs > max_fs:
                            max_fs = fs
                            best_k = k
                        elif fs == max_fs and k < best_k:  # Tie-breaking for consistency
                            best_k = k
            if best_k != -1 and max_fs >= self.hyper_params.vigilance:
                local_assignment[i] = best_k
                element_max_score[i] = max_fs
        # print(f\"    Local Winner Assignment: {local_assignment}\")
        
        # --- 5b: Identification of Common Elements ---
        # Using a SIMPLIFIED version based only on the final_fs threshold
        common_elements_map = self._identify_common_elements_simplified(n_elements, final_preds)
        # print(f\"    Common Elements Map: {common_elements_map}\")
        
        # --- 5c: Formation of Raw Output and Stabilizer Call ---
        stabilized_output_scores: List[Tuple[Any, Tuple[int, float]]] = []
        current_window_stable_ids: List[int] = []  # For updating Stabilizer transitions
        print("    Getting stabilized output from Stabilizer...")
        for i in range(n_elements):
            symbol = original_chars[i]
            local_winner_id = local_assignment[i]
            local_winner_score = element_max_score[i]
            
            # --- Form raw signature vector for element i ---
            raw_signature_vector_i: List[float] = [0.0] * len(self.column_ids)
            col_id_map = {col_id: idx for idx, col_id in enumerate(self.column_ids)}  # Map ID to vector index
            
            raw_signature_vector_i: List[float] = [0.0] * len(self.column_ids)
            max_p_single_for_context = -1.0
            context_col_id = -1  # ID of the column with max p_single for this element i
            
            if final_preds:
                for k, pred_list in final_preds.items():
                    list_idx = col_id_map.get(k)
                    if list_idx is None:
                        continue  # Should not happen if col_id_map is from self.column_ids
                    
                    if pred_list and i < len(pred_list) and isinstance(pred_list[i], dict):
                        p_single_val = pred_list[i].get('p_single')
                        if isinstance(p_single_val, (int, float)) and math.isfinite(p_single_val):
                            current_p_single = max(0.0, min(1.0, p_single_val))
                            if list_idx < len(raw_signature_vector_i):  # Check index bounds
                                raw_signature_vector_i[list_idx] = current_p_single
                                # Search for the column with maximum p_single
                                if current_p_single > max_p_single_for_context:
                                    max_p_single_for_context = current_p_single
                                    context_col_id = k  # Remember column ID k
            
            # +++ START OF SIGNATURE DEBUGGING BLOCK +++
            # Print only for key symbols for analysis
            symbols_to_debug = ['a', 'b', ' ', 'c', 's', 'з', 'A']  # Add/remove symbols as needed
            if symbol in symbols_to_debug:
                # print(f"    CM DEBUG (Step {current_processing_step}, Elem {i}, Sym '{symbol}'):")
                sig_array = np.array(raw_signature_vector_i)
                # Print the signature itself (can be rounded for readability)
                print(
                    f"      Raw Signature: {np.array2string(sig_array, precision=3, suppress_small=True)}")
                # Print indices and values of top 3 active columns
                top_indices = np.argsort(sig_array)[-3:][::-1]  # Indices of the 3 maximum
                top_values = sig_array[top_indices]
                print(
                    f"      Top 3 Raw FS: { {self.column_ids[k_idx]: f'{v:.3f}' for k_idx, v in zip(top_indices, top_values) if k_idx < len(self.column_ids)} }")
            # +++ END OF SIGNATURE DEBUGGING BLOCK +++
            
            # --- Call the correct Stabilizer method with the vector ---
            stable_id_i = self.stabilizer.get_stable_cluster_id(
                symbol,
                raw_signature_vector_i,  # Pass the raw signature vector
                is_window_novel=is_novel_window,
                context_col_id=context_col_id
            )
            # -------------------------------------------------------
            current_window_stable_ids.append(stable_id_i)  # Save for updating transitions
            
            # Form the final ID for output (Priority to set ID if it MATCHES the stable one?).
            # Or just use StableID? Yes, as Stabilizer has already considered consensus.
            final_output_id = stable_id_i
            final_output_score = local_winner_score  # Use the local winner's score
            
            stabilized_output_scores.append((symbol, (final_output_id, final_output_score)))
        
        # --- 5d: Updating Transition Statistics in Stabilizer ---
        # This step is now FULLY encapsulated within the get_stable_cluster_id calls,
        # as it updates self.previous_stable_id internally with each call.
        print(f"    Stabilizer transition stats were updated internally during get_stable_cluster_id calls.")
        
        # --- 5e: Determine overall_winner_id ---
        # Base it on the FIRST element of the STABILIZED output
        overall_winner_id = stabilized_output_scores[0][1][0] if stabilized_output_scores else -1
        
        # --- 5f: Update MaxIdealness history ---
        if not is_novel_window:
            self._update_max_idealness_history(max_idealness)
        
        print(f"  Finished Step 5. Final Output (stabilized, sample): {stabilized_output_scores[:5]}...")
        print(f"--- Processing Finished (Emergent + Stabilizer). Overall Winner: {overall_winner_id} ---")
        return overall_winner_id, stabilized_output_scores  # OUTPUT
    
    # === NEW METHOD FOR VISUALISATION ===
    def plot_dynamics(self, save_dir: Optional[Path] = None):
        """
        Plots and saves (if path is specified) dynamics graphs
        of SumIdealness and Modulation Signal for all columns.
        """
        print(f"\nCM {self.name}: Generating dynamics plots...")
        if not self.history_steps or not self.columns:
            print("  WARN: Not enough history or no columns to plot.")
            return
        
        # --- Sum Idealness Graph ---
        try:
            plt.figure(figsize=(15, 7))
            num_columns = len(self.columns)
            # Use colormaps['viridis'] or another palette
            try:
                # Preferred method for matplotlib 3.x+
                palette = colormaps.get_cmap('tab10')
            except AttributeError:  # For older versions
                from matplotlib import cm as colormap_module
                palette = colormap_module.get_cmap('tab10')
                # -------------------------------------------
            colours = palette(np.linspace(0, 1, num_columns))  # Use the obtained palette
            
            col_ids_sorted = sorted(self.columns.keys())
            
            max_y_idealness = 0  # For automatic scaling
            
            for idx, col_id in enumerate(col_ids_sorted):
                metric_values = list(self.history_sum_idealness.get(col_id, []))  # Take from dictionary
                if len(metric_values) != len(self.history_steps):
                    print(f"    WARN (SumIdealness): Skipping Col {col_id} due to history length mismatch.")
                    continue
                metric_values_plot = [v if v is not None and math.isfinite(v) else np.nan for v in metric_values]
                plt.plot(self.history_steps, metric_values_plot, label=f'Col {col_id}',
                         color=colours[idx % len(colours)], marker='.', markersize=2, linewidth=1, alpha=0.8)
                # Update maximum for Y-axis
                valid_vals = [v for v in metric_values_plot if not np.isnan(v)]
                if valid_vals:
                    max_y_idealness = max(max_y_idealness, max(valid_vals))
            
            plt.xlabel("Processing Step")
            plt.ylabel("Sum Idealness")
            plt.title(f"CM {self.name}: Column Sum Idealness Dynamics")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.ylim(bottom=-0.1, top=max_y_idealness * 1.1 if max_y_idealness > 0 else 1.0)  # Auto-scale Y
            plt.tight_layout(rect=(0, 0, 0.85, 1))
            
            if save_dir:
                save_path = save_dir / f"cm_{self.name}_sum_idealness_dynamics.png"
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150)
                plt.close()
                print(f"  Sum Idealness plot saved to {save_path}")
            else:
                plt.show()
        
        except Exception as e:
            print(f"ERROR plotting Sum Idealness: {e}")
            traceback.print_exc()
            plt.close()
        
        # --- Modulation Signal Graph ---
        try:
            plt.figure(figsize=(15, 7))
            # Same colours
            for idx, col_id in enumerate(col_ids_sorted):
                metric_values = list(self.history_modulation.get(col_id, []))
                if len(metric_values) != len(self.history_steps):
                    print(f"    WARN (Modulation): Skipping Col {col_id} due to history length mismatch.")
                    continue
                metric_values_plot = [v if v is not None and math.isfinite(v) else np.nan for v in metric_values]
                plt.plot(self.history_steps, metric_values_plot, label=f'Col {col_id}',
                         color=colours[idx % len(colours)], marker='.', markersize=2, linewidth=1, alpha=0.8)
            
            plt.xlabel("Processing Step")
            plt.ylabel("Modulation Signal")
            plt.title(f"CM {self.name}: Column Modulation Signal Dynamics")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.ylim(bottom=-0.05, top=1.05)  # Fixed range for modulation
            plt.tight_layout(rect=(0, 0, 0.85, 1))
            
            if save_dir:
                save_path = save_dir / f"cm_{self.name}_modulation_dynamics.png"
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150)
                plt.close()
                print(f"  Modulation Signal plot saved to {save_path}")
            else:
                plt.show()
        
        except Exception as e:
            print(f"ERROR plotting Modulation Signal: {e}")
            traceback.print_exc()
            plt.close()
    
    def dump(self):
        print(f"ColumnManager: Saving state ({len(self.columns)} column(s))...")
        save_errors = 0
        try:
            # Saving metadata state
            self.metadata.data.update({
                attr: getattr(self, attr)
                for attr in dir(self)
                if attr in self.metadata.data.keys() and hasattr(self, attr)
            })
            self.metadata.dump()
            
            # Saving namers state
            self.input_namer.dump()
            self.column_namer.dump()
            
            # Saving column states
            for column in self.columns.values():
                try:
                    print(f'    Dumping column {column.name} state...')
                    column.dump()
                
                except Exception as e:
                    print(f'    Could not dump {column.name} state!')
            
            # Saving stabilizer state
            self.stabilizer.dump()
        
        except Exception as e:
            print(f"Error saving manager metadata: {e}")
            save_errors += 1
        status = "successfully" if save_errors == 0 else f"with {save_errors} errors"
        print(f"ColumnManager: State save finished {status}.")


if __name__ == '__main__':
    from sliding_window import sliding_window
    
    INPUT_WIDTH = 18
    WINDOW_STEP = 1
    NUM_TRAIN_WINDOWS_RU = 200
    NUM_TRAIN_WINDOWS_EN = 200
    COLUMNS_QUANTITY = 8
    
    c_manager = ColumnManagerEmergent(0, Path('./data/c_managers'), columns_quantity=COLUMNS_QUANTITY,
                                      input_width=INPUT_WIDTH)
    
    with open(Path('./datasets/early/ru/ru_dict_nl.txt'), 'r', encoding='utf-8') as file:
        TEXT_DATA_RU = file.read()
    
    with open(Path('./datasets/early/ru/ru_dict_nl.txt'), 'r', encoding='utf-8') as file:
        TEXT_DATA_RU_NL = file.read()
    
    with open(Path('./datasets/early/ru/ru_spaced_dict.txt'), 'r', encoding='utf-8') as file:
        TEXT_DATA_EN = file.read()
    
    with open(Path('./datasets/early/gr/gr_spaced_dict.txt'), 'r', encoding='utf-8') as file:
        TEXT_DATA_GR = file.read()
    
    training_windows_ru = []
    training_windows_en = []
    training_windows_gr = []
    window_generator_ru = sliding_window(TEXT_DATA_RU, INPUT_WIDTH, WINDOW_STEP)
    window_generator_ru_nl = sliding_window(TEXT_DATA_RU_NL, INPUT_WIDTH, WINDOW_STEP)
    window_generator_en = sliding_window(TEXT_DATA_EN, INPUT_WIDTH, WINDOW_STEP)
    window_generator_gr = sliding_window(TEXT_DATA_GR, INPUT_WIDTH, WINDOW_STEP)
    
    # --- Training on Sliding Window ---
    print(f"\n--- Training on {NUM_TRAIN_WINDOWS_RU} windows ---")
    
    # for wi, window in enumerate(window_generator_ru_nl):
    #     if wi >= NUM_TRAIN_WINDOWS:
    #         break
    #
    #     cleaned_input = list(window)
    #
    #     print(wi, window)
    #     c_manager.process(window)
    
    # Train on Cyrillic
    for i, window_tuple in enumerate(window_generator_ru):
        if i >= NUM_TRAIN_WINDOWS_RU:
            break
        window_str = "".join(window_tuple)
        training_windows_ru.append(window_tuple)  # Save for prediction
        print(f"Training window {i + 1}/{NUM_TRAIN_WINDOWS_RU}: '{window_str}'")
        
        # Prepare data for the column
        input_data = [c for c in window_tuple]
        
        if len(input_data) < 2:  # Skip windows that are too short (due to remainder)
            print("  Window too short, skipping.")
            continue
        
        try:
            response = c_manager.process(input_data)
            print(f'DEBUG: Response: {response}')
        except Exception as e:
            print(f"  Error training on window {window_tuple}: {e}")
            # Continue with the next window
    
    # --- Training on Sliding Window ---
    print(f"\n--- Training on {NUM_TRAIN_WINDOWS_EN} windows ---")
    # Train on Latin
    for i, window_tuple in enumerate(window_generator_en):
        if i >= NUM_TRAIN_WINDOWS_EN:
            break
        window_str = "".join(window_tuple)
        training_windows_en.append(window_tuple)  # Save for prediction
        print(f"Training window {i + 1}/{NUM_TRAIN_WINDOWS_EN}: '{window_str}'")
        
        # Prepare data for the column
        input_data = [c for c in window_tuple]
        
        if len(input_data) < 2:  # Skip windows that are too short (due to remainder)
            print("  Window too short, skipping.")
            continue
        
        try:
            response = c_manager.process(input_data)
            print(f'DEBUG: Response: {response}')
        except Exception as e:
            print(f"  Error training on window: {e}")
            # Continue with the next window
    
    # Train on Greek
    # for i, window_tuple in enumerate(window_generator_gr):
    #     if i >= NUM_TRAIN_WINDOWS:
    #         break
    #     window_str = "".join(window_tuple)
    #     training_windows_gr.append(window_tuple)  # Save for prediction
    #     print(f"Training window {i + 1}/{NUM_TRAIN_WINDOWS}: '{window_str}'")
    #
    #     # Prepare data for the column
    #     input_data = [c for c in window_tuple]
    #
    #     if len(input_data) < 2:  # Skip windows that are too short (due to remainder)
    #         print("  Window too short, skipping.")
    #         continue
    #
    #     try:
    #         response = c_manager.process(input_data)
    #         print(f'DEBUG: Response: {response}')
    #     except Exception as e:
    #         print(f"  Error training on window: {e}")
    #         # Continue with the next window
    
    # Presenting a test window
    test_window_str = 'абто ant'
    test_window = list(test_window_str)[:INPUT_WIDTH]
    
    response = c_manager.process(test_window)
    print(test_window, len(test_window))
    print('Test Response', response)
    
    # Visualisation of Stabilizer centroids
    # try:
    #     stabilizer_plot_path = Path('./results/early/plots_data') / f"stabilizer_{c_manager.stabilizer.name}_centroids.png"
    #     c_manager.stabilizer.plot_centroid_heatmap(save_path=stabilizer_plot_path)
    # except Exception as e:
    #     print(f"Error generating stabilizer plot: {e}")
    
    # c_manager.plot_dynamics(Path('./results/early/plots_data'))
    
    c_manager.dump()