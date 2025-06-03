import streamlit as st
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, Any, Optional

from src.core.fep_ph_clasterer import LifelongClusterer

def run_clusterer_streamlit():
    
    global dataset
    with open("./datasets/fep_ph_clusterer/plain_strings.txt", 'r', encoding='utf-8') as dataset_file:
        dataset = dataset_file.readlines()
    
    st.set_page_config(layout="wide")
    st.title("🧐 HELOS Clusterer Visualisation")
    
    st.markdown("""
    This tool visualises the structure and state of the **HELOS LifelongClusterer**.

    You can either:
    - 🛠️ *Generate a clusterer state* using predefined sample data.
    - 📁 *Upload a saved clusterer JSON* generated from another run.
    """)
    
    # --- State Management ---
    clusterer_state: Optional[Dict[str, Any]] = None
    
    # --- Generate from sample data ---
    if st.button("🛠️ Generate Clusterer State from Sample Data"):
        clusterer = LifelongClusterer(
            stability_threshold=0.5,
            persistence_history_length=20,
            required_stability_passes=2,
            min_cluster_size=5,
            min_cooccurrence_for_ph=1
        )
        
        for text in dataset:
            clusterer.process_string(text)
        
        clusterer_state = clusterer.get_state()
        st.session_state["clusterer_state"] = clusterer_state
    
    # --- Upload JSON file ---
    uploaded_file = st.file_uploader("📁 Upload Clusterer State (JSON)", type="json")
    if uploaded_file:
        clusterer_state = json.load(uploaded_file)
        st.session_state["clusterer_state"] = clusterer_state
    
    # --- Load from memory if exists ---
    if "clusterer_state" in st.session_state:
        clusterer_state = st.session_state["clusterer_state"]
    
    # --- Main Visualisation ---
    if clusterer_state:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔗 Co-occurrence Graph")
            G = nx.Graph()
            
            # Assign unique colour to each cluster
            cluster_palette = cm.get_cmap('tab10', len(clusterer_state["cluster_ids"]))
            cluster_id_to_colour = {}
            for i, cid in enumerate(sorted(clusterer_state["cluster_ids"])):
                cluster_id_to_colour[cid] = cluster_palette(i)
            
            node_colors = {}
            
            # Add cluster nodes and their members
            for cluster_id in clusterer_state["cluster_ids"]:
                cluster_id_str = cluster_id
                cluster_colour = cluster_id_to_colour[cluster_id_str]
                G.add_node(cluster_id_str, type="cluster")
                node_colors[cluster_id_str] = cluster_colour
                
                definition = clusterer_state["cluster_definitions"].get(cluster_id_str, [])
                for member in definition:
                    member_str = member
                    G.add_node(member_str, type="symbol")
                    G.add_edge(cluster_id_str, member_str, weight=1)
                    node_colors[member_str] = cluster_colour
            
            # Add remaining symbol nodes
            for symbol in clusterer_state["symbols"]:
                symbol_str = symbol
                if symbol_str not in G:
                    G.add_node(symbol_str, type="symbol")
                    node_colors[symbol_str] = "skyblue"
            
            # Add co-occurrence edges
            for node1, neighbours in clusterer_state["cooccurrence_counts"].items():
                node1_str = node1
                for node2, count in neighbours.items():
                    node2_str = node2
                    if node1_str != node2_str:
                        G.add_edge(node1_str, node2_str, weight=count)
            
            # Draw
            plt.figure(figsize=(12, 10))
            k_val = 2 / (len(G.nodes) ** .2)
            pos = nx.spring_layout(G, seed=42, k=k_val, iterations=100)
            node_color_list = [node_colors.get(n, "grey") for n in G.nodes()]
            nx.draw(G, pos, with_labels=True, node_size=800, node_color=node_color_list, edge_color="grey")
            st.pyplot(plt)
        
        with col2:
            st.subheader("📦 Clusters")
            for cid, members in clusterer_state["cluster_definitions"].items():
                st.markdown(f"**Cluster {cid}**: {', '.join(members)}")
        
        st.subheader("📈 Co-occurrence Details")
        selected_node = st.selectbox("Select a node to inspect",
                                     sorted(clusterer_state["cooccurrence_counts"].keys(), key=str))
        if selected_node:
            st.json(clusterer_state["cooccurrence_counts"][selected_node])