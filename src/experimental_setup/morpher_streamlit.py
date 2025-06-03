# SPDX-License-Identifier: LicenseRef-NonCommercial-Research
# Copyright (c) 2025 Alexei Firssoff. ORCID: 0009-0006-0316-116X

"""
HELOS: Hierarchical Emergence of Latent Ontological Structure
Author: Alexei Firssoff
License: See LICENSE.txt for terms and conditions
"""

import os
import sys
import pandas as pd
import streamlit as st
import altair as alt

# Add project root to sys.path to enable relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import experiment setup functions from morpher module
from src.experimental_setup.morpher_setup import (
    prepare_data,
    train_morfessor,
    analyze_test_set,
    compare_results,
    load_gold_segments
)


def run_morpher_streamlit():
    """
    Launches a Streamlit app to compare morphological segmentation quality
    between HELOS and Morfessor on small language corpora.

    Allows user to:
    - Select language
    - Run training and evaluation
    - View and compare F1 scores
    - Inspect HELOS trees and segmentations for specific words
    """
    
    # Mapping from display language name to language code
    lang_options = {
        "Russian": "rus",
        "Turkish": "tur",
        "French": "fra",
        "German": "deu"
    }
    
    # Initialise session state with default language
    if "lang_selected" not in st.session_state:
        st.session_state.lang_selected = "Russian"
    
    # Language selector dropdown
    lang_display = st.selectbox("🌍 Select language:", list(lang_options.keys()),
                                index=list(lang_options.keys()).index(st.session_state.lang_selected))
    st.session_state.lang_selected = lang_display
    
    # Get language code
    LANG = lang_options[lang_display]
    
    # Define file paths for corpus, model, and results
    SMALL_CORPUS_FILE = f"./datasets/dicts/{LANG}/small_corpus.txt"
    SMALL_TRAIN_FILE = f"./datasets/dicts/{LANG}/small_train.txt"
    SMALL_TEST_FILE = f"./datasets/dicts/{LANG}/small_test.txt"
    MORPHEMES_BENCHMARK = f"./datasets/dicts/{LANG}/morphemes_benchmark.txt"
    MORFESSOR_MODEL_FILE = f"./results/temp/morfessor_model_{LANG}.bin"
    RESULTS_CSV = f"./results/fep_morpher/eval_vs_gold_{LANG}.csv"
    
    # 📃 Load results CSV with caching
    @st.cache_data
    def load_data(lang):
        return pd.read_csv(RESULTS_CSV)
    
    # 💡 Main training + evaluation logic (triggered by button)
    if st.button("🚀 Run training & comparison"):
        
        if not os.path.exists(SMALL_CORPUS_FILE):
            st.error("Please create 'small_corpus_[lang].txt' with ~100 words, one per line.")
        else:
            # Prepare train/test sets
            train_words, test_words = prepare_data(
                SMALL_CORPUS_FILE, SMALL_TRAIN_FILE, SMALL_TEST_FILE, test_ratio=0.2
            )
            
            print("train test", train_words, test_words)
            
            # Train and evaluate models if data was prepared successfully
            if train_words and test_words:
                morfessor_trained = train_morfessor(SMALL_TRAIN_FILE, MORFESSOR_MODEL_FILE)
                gold_segments = load_gold_segments(MORPHEMES_BENCHMARK)
                helos_results, morfessor_results = analyze_test_set(test_words, MORFESSOR_MODEL_FILE)
                compare_results(helos_results, morfessor_results, test_words, gold_segments, RESULTS_CSV)
                st.success(f"Done! Results saved to {RESULTS_CSV}")
                
                df = load_data(LANG)
            else:
                st.error("Could not prepare data.")
    
    # If results CSV exists, visualise comparison
    if os.path.exists(RESULTS_CSV):
        df = load_data(RESULTS_CSV)
        
        # Split full and summary data
        word_df = df[df['Word'] != '— AVERAGE —']
        summary_df = df[df['Word'] == '— AVERAGE —']
        
        # App header and evaluation warning
        st.title("🔍 HELOS vs Morfessor: Morphological Segmentation Evaluation")
        st.markdown(
            """
            ⚠️ **Important note on evaluation accuracy:**
            The current F1 evaluation for HELOS is provisional, due to limitations in the method used to convert tree-structured outputs into linear segmentations.
            This mapping does not yet fully capture the intended morphological structure inferred by the model. An improved alignment procedure is under development.
            In the meantime, users are encouraged to rely on the visualised HELOS parses for qualitative assessment and direct comparison with the gold and Morfessor outputs.
            """,
            unsafe_allow_html=True
        )
        
        # Summary table of average metrics
        st.subheader("📈 Average Performance Metrics")
        st.dataframe(summary_df.set_index('Word'), use_container_width=True)
        
        # Parse F1 scores for plotting
        word_df["F1 (HELOS)"] = pd.to_numeric(word_df["F1 (HELOS)"], errors="coerce")
        word_df["F1 (Morfessor)"] = pd.to_numeric(word_df["F1 (Morfessor)"], errors="coerce")
        
        # Prepare bar chart data
        f1_df = word_df[["Word", "F1 (HELOS)", "F1 (Morfessor)"]].dropna()
        f1_chart_data = f1_df.melt(id_vars=["Word"],
                                   value_vars=["F1 (HELOS)", "F1 (Morfessor)"],
                                   var_name="Model", value_name="F1")
        
        # F1 bar chart
        f1_chart = alt.Chart(f1_chart_data).mark_bar().encode(
            x=alt.X("Word:N", sort="-y", title="Word"),
            y=alt.Y("F1:Q", title="F1 Score"),
            color=alt.Color("Model:N", title="Model"),
            tooltip=["Word", "Model", "F1"]
        ).properties(width=700, height=400)
        
        st.altair_chart(f1_chart, use_container_width=True)
        
        # Select a word to inspect in detail
        selected_word = st.selectbox("🔍 Select a word to inspect:", word_df['Word'].tolist())
        selected_row = word_df[word_df['Word'] == selected_word].iloc[0]
        
        st.markdown(f"### 🔠 Morphological Analysis for **{selected_word}**")
        
        # Display HELOS tree
        st.write("**HELOS Tree:**")
        st.code(selected_row['HELOS Morphs'], language="text")
        
        # Display Morfessor segmentation
        st.write("**Morfessor Segmentation:**")
        st.code(selected_row['Morfessor Morphs'], language="text")
        
        # Display Gold segmentation
        st.write("**Gold Segmentation:**")
        st.code(selected_row['Gold Morphs'], language="text")
        
        # Precision/Recall/F1 comparison table
        st.subheader("📋 Metrics")
        metrics_data = {
            "": ["Precision", "Recall", "F1-score"],
            "HELOS": [
                selected_row['Precision (HELOS)'],
                selected_row['Recall (HELOS)'],
                selected_row['F1 (HELOS)']
            ],
            "Morfessor": [
                selected_row['Precision (Morfessor)'],
                selected_row['Recall (Morfessor)'],
                selected_row['F1 (Morfessor)']
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.set_index(""), use_container_width=True)
    
    else:
        st.warning("Evaluation file not found. Please run training and comparison first.")
