# SPDX-License-Identifier: LicenseRef-NonCommercial-Research
# Copyright (c) 2025 Alexei Firssoff. ORCID: 0009-0006-0316-116X

"""
HELOS: Hierarchical Emergence of Latent Ontological Structure
Author: Alexei Firssoff
License: See LICENSE.txt for terms and conditions
"""

import random
import os
import sys
import time
import morfessor
import subprocess
import pandas as pd
from IPython.display import display
from src.core.fep_morpher import Node, process_word_fep_v3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# --- Settings ---
HELOS_EPOCHS = 50           # Number of training epochs for HELOS
HELOS_BEAM_TRAIN = 20       # Beam width during HELOS training
HELOS_BEAM_TEST = 20        # Beam width during HELOS testing


# --- 1. Data Preparation (train/test split) ---
def prepare_data(corpus_file, train_file, test_file, test_ratio=0.2):
    """
    Splits the input corpus into training and test sets and writes them to separate files.

    :param corpus_file: Path to the full word corpus.
    :param train_file: Path to save the training words.
    :param test_file: Path to save the test words.
    :param test_ratio: Proportion of the corpus to use for testing.
    :return: Tuple (train_words, test_words)
    """
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
        random.shuffle(words)
        split_idx = int(len(words) * (1 - test_ratio))
        train_words = words[:split_idx]
        test_words = words[split_idx:]
        with open(train_file, "w", encoding="utf-8") as f:
            for word in train_words:
                f.write(word + "\n")
        with open(test_file, "w", encoding="utf-8") as f:
            for word in test_words:
                f.write(word + "\n")
        print(f"Data prepared: {len(train_words)} train, {len(test_words)} test.")
        return train_words, test_words
    except FileNotFoundError:
        print(f"Error: Corpus file '{corpus_file}' not found.")
        return [], []


# --- 2. HELOS Training ---
def train_helos(train_words):
    """
    Trains the HELOS model on the provided training words using beam search.

    :param train_words: List of words to use for training.
    """
    global node_registry, node_creation_counter
    node_registry = {}
    node_creation_counter = 0
    print("\n--- Training HELOS ---")
    start_train_time = time.time()
    for epoch in range(HELOS_EPOCHS):
        print(f"Epoch {epoch + 1}/{HELOS_EPOCHS}...")
        random.shuffle(train_words)
        for word in train_words:
            process_word_fep_v3(word, update_weights=True, top_k=2, beam_width=HELOS_BEAM_TRAIN,
                                visualize=False)  # No visualisation during training
    end_train_time = time.time()
    print(
        f"HELOS training finished in {end_train_time - start_train_time:.2f} sec. Registry size: {len(node_registry)}")


# --- 3. Morfessor Training ---
def train_morfessor(train_file, model_file):
    """
    Trains Morfessor using a command-line interface and saves the model.

    :param train_file: Path to training data.
    :param model_file: Prefix for the model output.
    :return: True if training succeeded, False otherwise.
    """
    print("\n--- Training Morfessor ---")
    txt_model_file = model_file + ".txt"
    try:
        command = [
            "morfessor-train",
            train_file,
            "-s", txt_model_file,
            "--encoding", "utf-8",
            "--corpusweight", "1.0"
        ]
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        print("Morfessor training complete.")
        print("Morfessor stdout:\n", result.stdout)
        print("Morfessor stderr:\n", result.stderr)
        return True
    except FileNotFoundError:
        print("Error: Morfessor script not found. Is 'morfessor' installed and in PATH?")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error during Morfessor training (return code {e.returncode}):")
        print("Morfessor stdout:\n", e.stdout)
        print("Morfessor stderr:\n", e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error training Morfessor: {e}")
        return False


# --- 4. Evaluation on Test Words ---
def analyze_test_set(test_words, model_file):
    """
    Runs HELOS and Morfessor on the test set and returns segmentation results.

    :param test_words: List of words to analyse.
    :param model_file: Path prefix to the trained Morfessor model.
    :return: Tuple (helos_results, morfessor_results)
    """
    helos_results = {}
    morfessor_results = {}

    print("\n--- Analyzing with HELOS ---")
    for word in test_words:
        hypotheses = process_word_fep_v3(word, update_weights=False, top_k=1, beam_width=HELOS_BEAM_TEST,
                                         visualize=False)
        if hypotheses:
            helos_results[word] = hypotheses[0].node

    print("\n--- Analyzing with Morfessor ---")
    txt_model_file = model_file + ".txt"
    try:
        io = morfessor.MorfessorIO(encoding='utf-8')
        morfessor_model = io.read_any_model(txt_model_file)
        print(f"Morfessor text model loaded from {txt_model_file}.")
        for word in test_words:
            morphs, _ = morfessor_model.viterbi_segment(word)
            morfessor_results[word] = morphs
    except FileNotFoundError:
        print(f"Error: Morfessor model file '{txt_model_file}' not found. Cannot segment.")
    except Exception as e:
        print(f"Error segmenting with Morfessor: {e}")

    return helos_results, morfessor_results


def load_gold_segments(path: str) -> dict:
    """
    Loads gold-standard segmentations from file.

    :param path: Path to the gold standard file.
    :return: Dictionary mapping word → list of morphemes.
    """
    gold_segments = dict({})
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or " " not in line:
                continue
            word, morphs_str = line.split(maxsplit=1)
            morphs = morphs_str.split("+")
            gold_segments[word] = morphs
    return gold_segments


def boundaries_from_morphemes(morphemes):
    """
    Converts a list of morphemes into a set of boundary indices.

    :param morphemes: List of morpheme strings.
    :return: Set of character indices representing boundaries.
    """
    boundaries = set()
    idx = 0
    for morph in morphemes[:-1]:
        idx += len(morph)
        boundaries.add(idx)
    return boundaries


def extract_helos_morphemes_as_strings(node, word):
    """
    Extracts linear morpheme strings from a HELOS node tree.

    :param node: Root node of the parse tree.
    :param word: Original word string.
    :return: List of morpheme substrings.
    """
    morphemes = []

    def helper(n, offset):
        if isinstance(n.content, str):
            return n.content, offset + 1

        left, right = n.content
        left_str, left_end = helper(left, offset)
        right_str, right_end = helper(right, left_end)

        morph = word[offset:right_end]
        morphemes.append(morph)

        return left_str + right_str, right_end

    helper(node, 0)
    return morphemes


def extract_helos_boundaries_by_tree(node):
    """
    Extracts boundary indices from a HELOS node tree.

    :param node: Root node of the parse tree.
    :return: Set of boundary positions.
    """
    boundaries = set()
    def helper(n, offset=0):
        if isinstance(n.content, str):
            return [n.content], offset
        left, right = n.content
        left_seq, _ = helper(left, offset)
        right_seq, _ = helper(right, offset + len(left_seq))
        boundaries.add(offset + len(left_seq))
        return left_seq + right_seq, offset
    helper(node)
    return boundaries


def compute_f1(predicted, gold):
    """
    Computes precision, recall, and F1 score.

    :param predicted: Set of predicted boundaries.
    :param gold: Set of gold-standard boundaries.
    :return: Tuple (precision, recall, f1)
    """
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def compare_results(helos_parses, morfessor_segments, test_words, gold_segments, output_path, save_csv: bool = True):
    """
    Compares the results of HELOS and Morfessor against gold-standard segmentations.

    :param helos_parses: Dictionary of HELOS outputs.
    :param morfessor_segments: Dictionary of Morfessor segmentations.
    :param test_words: List of test words.
    :param gold_segments: Gold-standard segmentations.
    :param output_path: Path to save the results CSV.
    :param save_csv: Whether to save the results.
    """
    print("\n--- Morphological Comparison (vs Gold Standard) ---")
    rows = []

    for word in test_words:
        helos_parse = helos_parses.get(word)
        morf_segs = morfessor_segments.get(word)
        gold_segs = gold_segments.get(word)

        if gold_segs is None:
            continue

        helos_repr = repr(helos_parse) if helos_parse else '—'
        helos_morphs = extract_helos_morphemes_as_strings(helos_parse, word)
        helos_boundaries = boundaries_from_morphemes(helos_morphs)
        morf_boundaries = boundaries_from_morphemes(morf_segs) if morf_segs else set()
        gold_boundaries = boundaries_from_morphemes(gold_segs)

        p_h, r_h, f1_h = compute_f1(helos_boundaries, gold_boundaries)
        p_m, r_m, f1_m = compute_f1(morf_boundaries, gold_boundaries)

        row = {
            'Word': word,
            'HELOS Morphs': helos_repr,
            'Morfessor Morphs': ' + '.join(morf_segs) if morf_segs else '—',
            'Gold Morphs': ' + '.join(gold_segs),
            'HELOS Segs': len(helos_boundaries) + 1 if helos_parse else 0,
            'Morfessor Segs': len(morf_segs) if morf_segs else 0,
            'Gold Segs': len(gold_segs),
            'Precision (HELOS)': round(p_h, 3),
            'Recall (HELOS)': round(r_h, 3),
            'F1 (HELOS)': round(f1_h, 3),
            'Precision (Morfessor)': round(p_m, 3),
            'Recall (Morfessor)': round(r_m, 3),
            'F1 (Morfessor)': round(f1_m, 3)
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(by='Word', inplace=True)

    avg_metrics = {
        'Word': '— AVERAGE —',
        'HELOS Morphs': '',
        'Morfessor Morphs': '',
        'Gold Morphs': '',
        'HELOS Segs': df['HELOS Segs'].mean(),
        'Morfessor Segs': df['Morfessor Segs'].mean(),
        'Gold Segs': df['Gold Segs'].mean(),
        'Precision (HELOS)': df['Precision (HELOS)'].mean(),
        'Recall (HELOS)': df['Recall (HELOS)'].mean(),
        'F1 (HELOS)': df['F1 (HELOS)'].mean(),
        'Precision (Morfessor)': df['Precision (Morfessor)'].mean(),
        'Recall (Morfessor)': df['Recall (Morfessor)'].mean(),
        'F1 (Morfessor)': df['F1 (Morfessor)'].mean()
    }

    df = pd.concat([df, pd.DataFrame([avg_metrics])], ignore_index=True)

    print("\n===== HELOS and Morfessor vs Gold =====\n")
    display(df)

    if save_csv:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"\n✅ Results saved: {output_path}")
        except Exception as e:
            print(f"Could not save csv: {e}")
