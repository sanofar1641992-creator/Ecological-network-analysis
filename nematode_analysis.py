import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from networkx.algorithms.community import girvan_newman, modularity

# --- 1. Load, Clean, and Prepare Data ---
# Load the CSV file.
df = pd.read_csv("Supplementary_Table-S1.csv")

# Clean column names by stripping any leading/trailing whitespace
df.columns = df.columns.str.strip()
df.rename(columns={'Buccal Cavity': 'Buccal_Cavity', 'AP&T': 'AP_T', 'A&N': 'A_N'}, inplace=True)
df.columns = df.columns.str.replace('&', '_').str.replace(' ', '_').str.strip()

# Correct the spelling inconsistency in the 'Tail' column
df['Tail'] = df['Tail'].replace('Clavate-conical Cylindrical', 'Clavate-conico-cylindrical')

# Drop rows with any missing values
df = df.dropna()

# Ensure 'Species' and 'Morphotype' columns are clean
df['Species'] = df['Species'].astype(str).str.strip()
df['Morphotype'] = df['Morphotype'].astype(str).str.strip()

# Define the regional columns for network analysis
regional_columns = ['WB', 'OR', 'AP_T', 'PY', 'TN', 'A_N', 'KL', 'KA', 'GA', 'MH', 'GJ', 'LD', 'O']

# --- Trait-to-Code Mapping ---
cuticle_map = {'Striated': '1', 'Punctated': '2', 'Desmen': '3', 'Smooth': '4'}
buccal_map = {'1A': '1', '1B': '2', '2A': '3', '2B': '4'}
amphid_map = {
    'Indistinct': '1', 'Pocket': '2', 'Loop': '3', 'Blister': '4', 'Spiral': '5',
    'Slit': '6', 'Circular': '7', 'Longitudinal slit': '8'
}
tail_map = {
    'Round': '1', 'Filiform': '2', 'Conical': '3',
    'Clavate-conico-cylindrical': '4', 'Conico-cylindrical': '5'
}

# Create a temporary DataFrame for generating new morphotype codes
temp_df = df.copy()
temp_df['Cuticle_Code'] = temp_df['Cuticle'].astype(str).map(cuticle_map)
temp_df['Buccal_Code'] = temp_df['Buccal_Cavity'].astype(str).map(buccal_map)
temp_df['Amphid_Code'] = temp_df['Amphid'].astype(str).map(amphid_map)
temp_df['Tail_Code'] = temp_df['Tail'].astype(str).map(tail_map)
temp_df = temp_df.dropna(subset=['Cuticle_Code', 'Buccal_Code', 'Amphid_Code', 'Tail_Code'])
temp_df['Morphotype_new'] = temp_df['Cuticle_Code'] + temp_df['Buccal_Code'] + temp_df['Amphid_Code'] + temp_df['Tail_Code']

# --- 2. Create the Morphotype-Region Presence Matrix ---
morphotype_presence_df = temp_df.groupby('Morphotype_new')[regional_columns].sum().ge(1).astype(int)

# --- 3. Build the Morphotype Co-occurrence Network with Jaccard Similarity ---
# Calculate Jaccard similarity matrix
morphotypes = morphotype_presence_df.index
num_morphotypes = len(morphotypes)
jaccard_matrix = np.zeros((num_morphotypes, num_morphotypes))

for i in range(num_morphotypes):
    for j in range(i, num_morphotypes):
        row_i = morphotype_presence_df.iloc[i, :].to_numpy()
        row_j = morphotype_presence_df.iloc[j, :].to_numpy()
        
        intersection = np.sum(row_i * row_j)
        union = np.sum(row_i + row_j) - intersection
        
        if union == 0:
            jaccard_matrix[i, j] = jaccard_matrix[j, i] = 0
        else:
            jaccard_matrix[i, j] = jaccard_matrix[j, i] = intersection / union

# Create a DataFrame from the Jaccard matrix
jaccard_df = pd.DataFrame(jaccard_matrix, index=morphotypes, columns=morphotypes)
np.fill_diagonal(jaccard_df.values, 0)

# Build the network graph from the Jaccard similarity matrix
G_cooccurrence = nx.from_pandas_adjacency(jaccard_df)
G_cooccurrence = nx.Graph(G_cooccurrence)
G_cooccurrence.remove_edges_from(nx.selfloop_edges(G_cooccurrence))
G_cooccurrence.remove_edges_from([(u, v) for u, v, d in G_cooccurrence.edges(data=True) if d['weight'] == 0])

# --- 4. Node Removal Simulation with 1000 Iterations ---
num_nodes = G_cooccurrence.number_of_nodes()
if num_nodes == 0:
    print("Network has no nodes. Cannot perform simulation.")
    exit()

# Targeted attack
betweenness_centrality = nx.betweenness_centrality(G_cooccurrence)
sorted_nodes_targeted = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)
G_targeted = G_cooccurrence.copy()
lcc_size_targeted = []

for i in range(num_nodes):
    if G_targeted.number_of_nodes() > 0:
        if G_targeted.number_of_edges() > 0:
            try:
                lcc_targeted = max(nx.connected_components(G_targeted), key=len)
                lcc_size_targeted.append(len(lcc_targeted) / num_nodes)
            except ValueError:
                lcc_size_targeted.append(0)
        else:
            lcc_size_targeted.append(G_targeted.number_of_nodes() / num_nodes)
        node_to_remove = sorted_nodes_targeted[i]
        G_targeted.remove_node(node_to_remove)
    else:
        lcc_size_targeted.append(0)
lcc_size_targeted.append(0)

# Random attack simulation with 1000 iterations
num_iterations = 1000
all_random_lcc_curves = []

for _ in range(num_iterations):
    G_random_temp = G_cooccurrence.copy()
    sorted_nodes_random_temp = list(G_random_temp.nodes())
    np.random.shuffle(sorted_nodes_random_temp)
    
    lcc_size_random_temp = []
    
    for i in range(num_nodes):
        if G_random_temp.number_of_nodes() > 0:
            if G_random_temp.number_of_edges() > 0:
                try:
                    lcc_random_temp = max(nx.connected_components(G_random_temp), key=len)
                    lcc_size_random_temp.append(len(lcc_random_temp) / num_nodes)
                except ValueError:
                    lcc_size_random_temp.append(0)
            else:
                lcc_size_random_temp.append(G_random_temp.number_of_nodes() / num_nodes)
            node_to_remove = sorted_nodes_random_temp[i]
            G_random_temp.remove_node(node_to_remove)
        else:
            lcc_size_random_temp.append(0)
    lcc_size_random_temp.append(0)
    all_random_lcc_curves.append(lcc_size_random_temp)

# Calculate the mean curve from all iterations
lcc_size_random = np.mean(all_random_lcc_curves, axis=0)
node_removed_percentage = np.linspace(0, 100, num_nodes + 1)

# --- 5. Find Collapse Thresholds ---
threshold = 0.4
targeted_collapse_threshold = -1
for i, size in enumerate(lcc_size_targeted):
    if size < threshold:
        targeted_collapse_threshold = (i-1) / num_nodes * 100
        break

random_collapse_threshold = -1
for i, size in enumerate(lcc_size_random):
    if size < threshold:
        random_collapse_threshold = (i-1) / num_nodes * 100
        break

# --- 6. Calculate Network-level Metrics for Table 4 ---
print("\n--- Calculating Network Metrics ---")
network_density = nx.density(G_cooccurrence)
avg_degree = np.mean([d for n, d in G_cooccurrence.degree()])

communities_generator = girvan_newman(G_cooccurrence)
try:
    best_community_partition = next(communities_generator)
    best_modularity = nx.community.modularity(G_cooccurrence, best_community_partition)
except StopIteration:
    best_modularity = 0

if nx.is_connected(G_cooccurrence):
    avg_path_length = nx.average_shortest_path_length(G_cooccurrence)
else:
    if G_cooccurrence.number_of_nodes() > 1:
        largest_cc = max(nx.connected_components(G_cooccurrence), key=len)
        subgraph = G_cooccurrence.subgraph(largest_cc)
        if subgraph.number_of_edges() > 0 and subgraph.number_of_nodes() > 1:
            avg_path_length = nx.average_shortest_path_length(subgraph)
        else:
            avg_path_length = "N/A (Disconnected or single node)"
    else:
        avg_path_length = "N/A (Single node network)"

# --- 7. Calculate Centrality Metrics for Tables 5 and 7 ---
degree_centrality = nx.degree_centrality(G_cooccurrence)
betweenness_centrality = nx.betweenness_centrality(G_cooccurrence)
closeness_centrality = nx.closeness_centrality(G_cooccurrence)

top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

keystone_morphotypes = [node for node, _ in top_betweenness]

# --- 8. Add Rare Connector Identification for Future Analysis ---
# Create a DataFrame for all centrality metrics
centrality_df = pd.DataFrame({
    'Degree Centrality': degree_centrality,
    'Betweenness Centrality': betweenness_centrality,
    'Closeness Centrality': closeness_centrality
})

# Identify rare connectors by looking for low degree but high betweenness
centrality_df['Degree_Rank'] = centrality_df['Degree Centrality'].rank(ascending=True)
centrality_df['Betweenness_Rank'] = centrality_df['Betweenness Centrality'].rank(ascending=False)

# Select top 5 rare connectors: lowest degree rank and highest betweenness rank
rare_connectors = centrality_df.sort_values(by=['Degree_Rank', 'Betweenness_Rank'], ascending=[True, False]).head(5)
rare_connectors = rare_connectors.reset_index().rename(columns={'index': 'Morphotype'})
rare_connector_nodes = rare_connectors['Morphotype'].tolist()

# --- 9. Plotting Both Figures in a Single Canvas ---
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# --- Plot A: Morphotype Co-occurrence Network with Keystone Species and Rare Connectors Highlighted ---
ax_network = axes[0]
if G_cooccurrence.number_of_edges() == 0:
    ax_network.text(0.5, 0.5, "Network has no edges.\nCannot be drawn.", ha='center', va='center', fontsize=12)
    ax_network.set_title("a. Morphotype Co-occurrence Network (Jaccard Similarity)", size=18, loc='left')
    ax_network.axis('off')
else:
    max_degree_centrality = max(degree_centrality.values()) if degree_centrality else 0
    if max_degree_centrality == 0:
        node_sizes = [500 for _ in G_cooccurrence.nodes()]
    else:
        node_sizes = [v * 10000 / max_degree_centrality for v in degree_centrality.values()]
    norm = mcolors.Normalize(vmin=0, vmax=max_degree_centrality)
    cmap = plt.cm.viridis
    node_colors = [cmap(norm(degree_centrality[node])) for node in G_cooccurrence.nodes()]
    pos = nx.spring_layout(G_cooccurrence, k=0.3, iterations=100, seed=42)

    nx.draw_networkx_nodes(G_cooccurrence, pos, nodelist=G_cooccurrence.nodes(),
                           node_size=node_sizes,
                           node_color=node_colors,
                           edgecolors='black',
                           linewidths=0.5,
                           ax=ax_network, alpha=0.8)

    # Highlight keystone morphotypes in red
    keystone_nodes = keystone_morphotypes
    keystone_node_sizes = [v * 10000 / max_degree_centrality for node, v in degree_centrality.items() if node in keystone_nodes]
    keystone_pos = {node: pos[node] for node in keystone_nodes}
    keystone_node_colors = [cmap(norm(degree_centrality[node])) for node in keystone_nodes]
    nx.draw_networkx_nodes(G_cooccurrence, keystone_pos, nodelist=keystone_nodes,
                           node_size=keystone_node_sizes,
                           node_color=keystone_node_colors,
                           edgecolors='red',
                           linewidths=2,
                           ax=ax_network, alpha=0.8)
    
    # Highlight rare connector morphotypes in yellow
    rare_connector_node_sizes = [v * 10000 / max_degree_centrality for node, v in degree_centrality.items() if node in rare_connector_nodes]
    rare_connector_pos = {node: pos[node] for node in rare_connector_nodes}
    rare_connector_node_colors = [cmap(norm(degree_centrality[node])) for node in rare_connector_nodes]
    nx.draw_networkx_nodes(G_cooccurrence, rare_connector_pos, nodelist=rare_connector_nodes,
                           node_size=rare_connector_node_sizes,
                           node_color=rare_connector_node_colors,
                           edgecolors='yellow',
                           linewidths=2,
                           ax=ax_network, alpha=0.8)

    nx.draw_networkx_edges(G_cooccurrence, pos, alpha=0.4, edge_color='gray', width=0.8, ax=ax_network)

    for node, (x, y) in pos.items():
        if node in keystone_nodes:
            color = 'red'
            fontweight = 'bold'
        elif node in rare_connector_nodes:
            color = 'yellow'
            fontweight = 'bold'
        else:
            color = 'black'
            fontweight = 'normal'
        ax_network.text(x, y, node, fontsize=8, color=color, ha='center', va='center', fontweight=fontweight)

    ax_network.set_title("a. Morphotype Co-occurrence Network (Jaccard Similarity)", size=18, loc='left')
    ax_network.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_network, orientation='vertical', fraction=0.02, pad=0.05)
    cbar.set_label("Degree Centrality (Co-occurrence)", fontsize=12)

# --- Plot B: Vulnerability Gradient ---
ax_vulnerability = axes[1]
ax_vulnerability.plot(node_removed_percentage, lcc_size_random, 'o-', label='Random Attack (1000 Iterations)', color='blue')
ax_vulnerability.plot(node_removed_percentage, lcc_size_targeted, 's-', label='Targeted Attack', color='red')
ax_vulnerability.set_title('b. Vulnerability Gradient: Network Resilience', fontsize=18, loc='left')
ax_vulnerability.set_xlabel('Percentage of Morphotypes Removed', fontsize=12)
ax_vulnerability.set_ylabel('Relative Size of Largest Connected Component', fontsize=12)
ax_vulnerability.grid(True, linestyle='--', alpha=0.6)
ax_vulnerability.legend()

plt.tight_layout()
output_filename = 'combined_network_analysis_jaccard_plot.png'
plt.savefig(output_filename, dpi=300)

# --- Create a DataFrame for the requested CSV file ---
centrality_df['Rank'] = centrality_df['Betweenness Centrality'].rank(ascending=False)
centrality_df = centrality_df.sort_values(by='Rank').reset_index().rename(columns={'index': 'Morphotype'})
centrality_df.to_csv('morphotype_centrality_metrics_jaccard.csv', index=False)
print("Centrality metrics saved to 'morphotype_centrality_metrics_jaccard.csv'")

# --- Print the requested tables ---
print("\nTable 4. Network-level metrics for the morphotype occurrence network (Jaccard), including density, modularity, average degree, and average path length.")
print("| Metric | Value |")
print("|:---|:---|")
print(f"| Density | {network_density:.4f} |")
print(f"| Modularity | {best_modularity:.4f} |")
print(f"| Average Degree | {avg_degree:.2f} |")
print(f"| Average Path Length | {avg_path_length} |")

print("\nTable 5. Top five morphotypes ranked by degree, betweenness, and closeness centrality in the morphotype occurrence network (Jaccard).")
print("| Rank | Degree Centrality | Betweenness Centrality | Closeness Centrality |")
print("|:---|:---|:---|:---|")
for i in range(5):
    deg_morphotype, deg_value = top_degree[i]
    bet_morphotype, bet_value = top_betweenness[i]
    clo_morphotype, clo_value = top_closeness[i]
    print(f"| {i+1} | {deg_morphotype} ({deg_value:.4f}) | {bet_morphotype} ({bet_value:.4f}) | {clo_morphotype} ({clo_value:.4f}) |")

print("\nTable 6. Resilience thresholds for morphotype occurrence network under targeted and random node removal scenarios (Jaccard).")
print("| Attack Scenario | Collapse Threshold (% of nodes removed) |")
print("|:---|:---|")
print(f"| Targeted Attack | {targeted_collapse_threshold:.2f} |")
print(f"| Random Attack | {random_collapse_threshold:.2f} |")

print("\nTable 7. Keystone Morphotypes (Top 5 by Centrality) (Jaccard)")
print("Keystone morphotypes are those with the highest centrality scores, indicating their critical role in the network's structure and connectivity. Nodes with high betweenness centrality are particularly important as 'bridges' between different groups.")
print("| Rank | Morphotype (Degree Centrality) | Morphotype (Betweenness Centrality) | Morphotype (Closeness Centrality) |")
print("|:---|:---|:---|:---|")
for i in range(5):
    deg_morphotype, deg_value = top_degree[i]
    bet_morphotype, bet_value = top_betweenness[i]
    clo_morphotype, clo_value = top_closeness[i]
    print(f"| {i+1} | {deg_morphotype} | {bet_morphotype} | {clo_morphotype} |")

print("\nTable 8. Rare Connectors (Relative Low Degree and High Betweenness)")
print("Rare connectors are morphotypes with low connectivity but a disproportionately high role as bridges in the network. This table identifies the top 5 morphotypes with the lowest degree centrality and highest betweenness centrality.")
print("| Rank | Morphotype | Degree Centrality | Betweenness Centrality |")
print("|:---|:---|:---|:---|")
for i, row in enumerate(rare_connectors.itertuples()):
    print(f"| {i+1} | {row.Morphotype} | {row._3:.4f} | {row._4:.4f} |")
