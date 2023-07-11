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
    'Adherence to Treatment',
    'Doctor-Patient Interaction',
    'Emotional Health',
    'Symptom Resolution',
    'Functional Status',
    'Health Literacy',
    'Cultural Competency',
    'Patient Engagement',
    'Health Disparities',
    'Patient Empowerment',
    'Patient-Centeredness',
    'Care Coordination',
    'Patient Safety',
    'Healthcare Policies'
])

# Add directed edges based on relationships
G.add_edges_from([
    ('Patient-Physician Relationship', 'Communication Skills'),
    ('Patient-Physician Relationship', 'Health Outcomes'),
    ('Patient-Physician Relationship', 'Continuity of Care'),
    ('Communication Skills', 'Information Giving'),
    ('Communication Skills', 'Information Seeking'),
    ('Communication Skills', 'Partnership Building'),
    ('Communication Skills', 'Rapport Building'),
    ('Communication Skills', 'Facilitation of Patient Expression'),
    ('Communication Skills', 'Doctor-Patient Interaction'),
    ('Health Outcomes', 'Symptom Resolution'),
    ('Health Outcomes', 'Functional Status'),
    ('Continuity of Care', 'Trust'),
    ('Continuity of Care', 'Patient Satisfaction'),
    ('Continuity of Care', 'Adherence to Treatment'),
    ('Information Giving', 'Patient Autonomy'),
    ('Information Giving', 'Shared Decision-Making'),
    ('Information Seeking', 'Patient Participation in Decision-Making'),
    ('Partnership Building', 'Rapport Building'),
    ('Partnership Building', 'Trust'),
    ('Rapport Building', 'Emotional Responsiveness'),
    ('Rapport Building', 'Empathy'),
    ('Facilitation of Patient Expression', 'Patient Satisfaction'),
    ('Facilitation of Patient Expression', 'Patient Participation in Decision-Making'),
    ('Doctor-Patient Interaction', 'Emotional Health'),
    ('Doctor-Patient Interaction', 'Health Literacy'),
    ('Doctor-Patient Interaction', 'Cultural Competency'),
    ('Doctor-Patient Interaction', 'Patient Engagement'),
    ('Doctor-Patient Interaction', 'Health Disparities'),
    ('Doctor-Patient Interaction', 'Patient Empowerment'),
    ('Doctor-Patient Interaction', 'Patient-Centeredness'),
    ('Doctor-Patient Interaction', 'Care Coordination'),
    ('Doctor-Patient Interaction', 'Patient Safety'),
    ('Doctor-Patient Interaction', 'Healthcare Policies')
])


# Perform community detection using Louvain algorithm
partition = community.best_partition(G.to_undirected())

# Draw the network graph with node colors based on communities and save as separate images
for i in range(max(partition.values()) + 1):
    # Filter nodes belonging to the current community
    community_nodes = [node for node, comm in partition.items() if comm == i]

    # Create a subgraph with the nodes in the current community
    subgraph = G.subgraph(community_nodes)

    # Draw the subgraph with node colors based on communities
    pos = nx.spring_layout(subgraph)
    nx.draw(subgraph, pos, with_labels=True, node_size=1000, cmap='tab20', edge_color='gray', arrowsize=12, font_size=8)

    # Save the network graph as an image
    filename = f"community_graph_{i}.png"
    plt.title(f"Community {i}")
    plt.axis('off')
    plt.savefig(filename)
    plt.clf()  # Clear the figure for the next iteration

    print(f"Saved graph for Community {i} as {filename}")
