import networkx as nx
import matplotlib.pyplot as plt
import community

# Create an empty directed graph
G = nx.DiGraph()

# Add nodes
G.add_nodes_from([
    'Patient-Physician Relationship',
    'Communication Skills',
    'Health Outcomes',
    'Continuity of Care',
    'Information Giving',
    'Information Seeking',
    'Partnership Building',
    'Rapport Building',
    'Facilitation of Patient Expression',
    'Patient Autonomy',
    'Patient Participation in Decision-Making',
    'Trust',
    'Shared Decision-Making',
    'Emotional Responsiveness',
    'Empathy',
    'Patient Satisfaction',
    'Adherence to Treatment'
])

# Add directed edges (relationships)
G.add_edges_from([
    ('Information Giving', 'Patient Autonomy'),
    ('Information Giving', 'Shared Decision-Making'),
    ('Information Seeking', 'Partnership Building'),
    ('Information Seeking', 'Shared Decision-Making'),
    ('Partnership Building', 'Rapport Building'),
    ('Partnership Building', 'Trust'),
    ('Partnership Building', 'Continuity of Care'),
    ('Rapport Building', 'Emotional Responsiveness'),
    ('Rapport Building', 'Empathy'),
    ('Facilitation of Patient Expression', 'Patient Satisfaction'),
    ('Facilitation of Patient Expression', 'Patient Participation in Decision-Making'),
    ('Patient Autonomy', 'Trust'),
    ('Patient Autonomy', 'Shared Decision-Making'),
    ('Patient Participation in Decision-Making', 'Shared Decision-Making'),
    ('Patient Participation in Decision-Making', 'Health Outcomes'),
    ('Trust', 'Patient Satisfaction'),
    ('Trust', 'Continuity of Care'),
    ('Continuity of Care', 'Patient Satisfaction'),
    ('Continuity of Care', 'Adherence to Treatment'),
    ('Continuity of Care', 'Trust')  # Additional directed edge
])

# Compute degree centrality
degree_centrality = nx.degree_centrality(G)

# Compute betweenness centrality
betweenness_centrality = nx.betweenness_centrality(G)

# Compute closeness centrality
closeness_centrality = nx.closeness_centrality(G)

# Compute eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(G)

# Compute PageRank
pagerank = nx.pagerank(G)

# Compute network density
network_density = nx.density(G)

# Compute degree assortativity
degree_assortativity = nx.degree_assortativity_coefficient(G)

# Perform community detection using Louvain algorithm
partition = community.best_partition(G.to_undirected())

# Print the results
print("Degree Centrality:", degree_centrality)
print("Betweenness Centrality:", betweenness_centrality)
print("Closeness Centrality:", closeness_centrality)
print("Eigenvector Centrality:", eigenvector_centrality)
print("PageRank:", pagerank)
print("Network Density:", network_density)
print("Degree Assortativity:", degree_assortativity)
print("Community Structure:", partition)

# Draw the network graph with node colors based on communities
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color=list(partition.values()), cmap='tab20', edge_color='gray', arrowsize=12, font_size=8)

# Show the network graph
plt.title("Directed Network with Clustering")
plt.axis('off')
plt.show()
