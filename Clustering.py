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

# Create a new graph based on the detected communities
community_graph = nx.Graph()
community_graph.add_nodes_from(G.nodes())
for u, v, data in G.edges(data=True):
    if partition[u] == partition[v]:
        community_graph.add_edge(u, v, **data)

# Draw the community-based network graph
pos = nx.spring_layout(community_graph)
nx.draw(community_graph, pos, with_labels=True, node_size=1000, node_color=list(partition.values()), cmap='tab20', edge_color='gray', arrowsize=12, font_size=8)

# Show the network graph
plt.title("Community-Based Network Graph")
plt.axis('off')
plt.show()
