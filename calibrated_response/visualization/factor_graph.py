import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Any, Tuple
from calibrated_response.models.variable import Variable
from calibrated_response.models.natural_response import NaturalEstimate
from calibrated_response.models.query import Estimate

def get_connected_variables(est: Estimate) -> List[str]:
    """Helper function to extract variable names from an Estimate object."""
    connected_vars = []
    if hasattr(est, 'variable'):
        connected_vars.append(est.variable)
    if hasattr(est, 'proposition'):
        connected_vars.append(est.proposition.variable)
    if hasattr(est, 'conditions'):
        for cond in est.conditions:
            connected_vars.append(cond.variable)
    return connected_vars

def plot_factor_graph(variables: List[Variable], estimates: List[Estimate], figsize: Tuple[int, int] = (14, 10), title: str = "Factor Graph") -> Tuple[nx.Graph, plt.Figure]:
    """
    Generates and plots an orderly factor graph from variables and estimates.
    Variables are represented as circular nodes, estimates as square factor nodes.
    
    Args:
        variables: List of variable objects.
        estimates: List of estimate objects.
        figsize: Tuple for the figure size.
        title: Title of the plot.
        
    Returns:
        Tuple containing the NetworkX Graph object and the Matplotlib Figure.
    """
    G = nx.Graph()
    
    var_nodes = []
    est_nodes = []
    
    # Add variable nodes
    for i, var in enumerate(variables):
        node_id = f"V_{i}"
        label = getattr(var, 'name', f"Var {i}")
        G.add_node(node_id, bipartite=0, type='variable', label=label)
        var_nodes.append(node_id)
        
    # Add estimate (factor) nodes and edges
    for i, est in enumerate(estimates):
        node_id = f"F_{i}"
        label = est.to_query_estimate()
        # label = f"Est {i}"
        # if hasattr(est, 'type'):
        #     label = str(est.type)
        # elif hasattr(est, 'name'):
        #     label = str(est.name)
            
        G.add_node(node_id, bipartite=1, type='estimate', label=label)
        est_nodes.append(node_id)
        
        # Determine which variables this estimate connects to
        connected_vars = get_connected_variables(est)
        
        for j, var in enumerate(variables):
            var_name = getattr(var, 'name', f"Var {j}")
            if var_name in connected_vars:
                G.add_edge(f"V_{j}", node_id)
                
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use bipartite layout with vertical alignment
    # This places variables in a column on the left, and estimates in a column on the right
    pos = nx.bipartite_layout(G, var_nodes, align='vertical')
    
    # Draw variable nodes (circles)
    nx.draw_networkx_nodes(G, pos, nodelist=var_nodes, node_color='#add8e6', 
                           node_shape='o', node_size=800, edgecolors='#333333', ax=ax)
    
    # Draw estimate nodes (squares)
    nx.draw_networkx_nodes(G, pos, nodelist=est_nodes, node_color='#90ee90', 
                           node_shape='s', node_size=800, edgecolors='#333333', ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color='#555555', ax=ax)
    
    # Draw labels outside the nodes for horizontal readability
    var_labels = {n: G.nodes[n]['label'] for n in var_nodes}
    est_labels = {n: G.nodes[n]['label'] for n in est_nodes}
    
    # Shift label positions outward
    # Variables on the left get shifted further left, Estimates on the right get shifted further right
    pos_var_labels = {k: (v[0] - 0.05, v[1]) for k, v in pos.items() if k in var_nodes}
    pos_est_labels = {k: (v[0] + 0.05, v[1]) for k, v in pos.items() if k in est_nodes}
    
    nx.draw_networkx_labels(G, pos_var_labels, labels=var_labels, font_size=10, 
                            font_weight='bold', horizontalalignment='right', ax=ax)
    nx.draw_networkx_labels(G, pos_est_labels, labels=est_labels, font_size=9, 
                            horizontalalignment='left', ax=ax)
    
    # Expand margins so the horizontal labels don't get cut off
    ax.margins(x=0.4)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    
    return G, fig

def plot_clustered_factor_graph(variables: List[Variable], estimates: List[Estimate], figsize: Tuple[int, int] = (14, 10), title: str = "Clustered Factor Graph") -> Tuple[nx.Graph, plt.Figure]:
    """
    Generates and plots a factor graph designed to show clustering of variables.
    Variables are colored and have a legend, estimates are unlabeled small squares.
    
    Args:
        variables: List of variable objects.
        estimates: List of estimate objects.
        figsize: Tuple for the figure size.
        title: Title of the plot.
        
    Returns:
        Tuple containing the NetworkX Graph object and the Matplotlib Figure.
    """
    G = nx.Graph()
    
    var_nodes = []
    est_nodes = []
    
    # Add variable nodes
    for i, var in enumerate(variables):
        node_id = f"V_{i}"
        label = getattr(var, 'name', f"Var {i}")
        G.add_node(node_id, bipartite=0, type='variable', label=label)
        var_nodes.append(node_id)
        
    # Add estimate (factor) nodes and edges
    for i, est in enumerate(estimates):
        node_id = f"F_{i}"
        G.add_node(node_id, bipartite=1, type='estimate')
        est_nodes.append(node_id)
        
        # Determine which variables this estimate connects to
        connected_vars = get_connected_variables(est)
        
        for j, var in enumerate(variables):
            var_name = getattr(var, 'name', f"Var {j}")
            if var_name in connected_vars:
                G.add_edge(f"V_{j}", node_id)
                
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use spring layout to show clustering
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Generate colors for variables
    cmap = plt.get_cmap('tab20')
    var_colors = [cmap(i % 20) for i in range(len(var_nodes))]
    
    # Draw variable nodes (circles)
    nx.draw_networkx_nodes(G, pos, nodelist=var_nodes, node_color=var_colors, 
                           node_shape='o', node_size=600, edgecolors='#333333', ax=ax)
    
    # Draw estimate nodes (small squares)
    nx.draw_networkx_nodes(G, pos, nodelist=est_nodes, node_color='#333333', 
                           node_shape='s', node_size=100, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.4, edge_color='#888888', ax=ax)
    
    # Create legend for variables
    import matplotlib.patches as mpatches
    legend_handles = []
    for i, var_node in enumerate(var_nodes):
        label = G.nodes[var_node]['label']
        patch = mpatches.Patch(color=var_colors[i], label=label)
        legend_handles.append(patch)
        
    # Add estimate to legend
    est_patch = mpatches.Patch(color='#333333', label='Estimates')
    legend_handles.append(est_patch)
        
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    
    return G, fig

def plot_variable_interaction_graph(variables: List[Variable], estimates: List[Estimate], figsize: Tuple[int, int] = (14, 10), title: str = "Variable Interaction Graph") -> Tuple[nx.Graph, plt.Figure]:
    """
    Generates and plots a unipartite graph of variables, where edges represent shared estimates.
    Node size is proportional to the total number of connected estimates.
    Edge thickness is proportional to the number of shared estimates.
    
    Args:
        variables: List of variable objects.
        estimates: List of estimate objects.
        figsize: Tuple for the figure size.
        title: Title of the plot.
        
    Returns:
        Tuple containing the NetworkX Graph object and the Matplotlib Figure.
    """
    B = nx.Graph()
    
    var_nodes = []
    est_nodes = []
    
    # Add variable nodes
    for i, var in enumerate(variables):
        node_id = f"V_{i}"
        label = getattr(var, 'name', f"Var {i}")
        B.add_node(node_id, bipartite=0, type='variable', label=label)
        var_nodes.append(node_id)
        
    # Add estimate (factor) nodes and edges
    for i, est in enumerate(estimates):
        node_id = f"F_{i}"
        B.add_node(node_id, bipartite=1, type='estimate')
        est_nodes.append(node_id)
        
        # Determine which variables this estimate connects to
        connected_vars = get_connected_variables(est)
        
        for j, var in enumerate(variables):
            var_name = getattr(var, 'name', f"Var {j}")
            if var_name in connected_vars:
                B.add_edge(f"V_{j}", node_id)
                
    # Project to unipartite graph of variables
    G = nx.bipartite.weighted_projected_graph(B, var_nodes)
    
    # Calculate node sizes based on degree in the original bipartite graph
    # (number of connected estimates)
    node_sizes = []
    base_size = 500
    size_multiplier = 200
    for node in G.nodes():
        degree = B.degree(node)
        node_sizes.append(base_size + degree * size_multiplier)
        
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use spring layout weighted by edge weight
    pos = nx.spring_layout(G, weight='weight', k=0.8, iterations=100, seed=42)
    
    # Generate colors for variables
    cmap = plt.get_cmap('tab20')
    var_colors = [cmap(i % 20) for i in range(len(var_nodes))]
    
    # Draw variable nodes
    nx.draw_networkx_nodes(G, pos, node_color=var_colors, 
                           node_shape='o', node_size=node_sizes, edgecolors='#333333', ax=ax)
    
    # Draw edges with thickness based on weight
    edges = G.edges(data=True)
    if edges:
        weights = [data['weight'] for u, v, data in edges]
        max_weight = max(weights) if weights else 1
        # Scale edge widths
        edge_widths = [1.0 + 4.0 * (w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='#555555', ax=ax)
    
    # Draw labels
    labels = {n: B.nodes[n]['label'] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    
    return G, fig