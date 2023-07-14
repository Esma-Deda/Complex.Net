import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
import PyPDF4
import community



# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


# Function to process the text and extract keywords
def process_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Removing Stop Words and unwanted tokens
    stop_words = set(stopwords.words('english'))
    excluded_words = ['tx', 'butowpn', 'md', 'streetjrr', 'thus']  # Add more excluded words here
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.isalpha() and len(token) > 1 and token.lower() not in excluded_words]

    return filtered_tokens


# Function to create co-occurrence network
def create_cooccurrence_network(keywords):
    # Create a co-occurrence network
    cooc_network = nx.Graph()

    # Add nodes to the network
    cooc_network.add_nodes_from(keywords)

    # Compute co-occurrence frequency and add edges to the network
    for word1, word2 in combinations(keywords, 2):
        if word2 in cooc_network[word1]:
            cooc_network[word1][word2]['weight'] += 1
        else:
            cooc_network.add_edge(word1, word2, weight=1)

    return cooc_network


# File path of the PDF article
file_path = '/Users/esmaisufi/Desktop/Articles/2.How-does-communication-heal--Pathways-linking-clinici_2009_Patient-Education.pdf'

# Extract text from the PDF
with open(file_path, 'rb') as pdf_file:
    pdf_reader = PyPDF4.PdfFileReader(pdf_file)
    article_text = ""
    for page in range(pdf_reader.getNumPages()):
        article_text += pdf_reader.getPage(page).extractText()

# Process the article text and extract keywords
keywords = process_text(article_text)

# Compute keyword frequency
keyword_counts = Counter(keywords)

# Get the 40 most important keywords
num_keywords = 40
top_keywords = [keyword for keyword, count in keyword_counts.most_common(num_keywords)]

# Create the co-occurrence network
cooc_network = create_cooccurrence_network(top_keywords)

# Define the patient-related terms
patient_terms = ['patient', 'symptoms', 'diagnosis', 'treatment', 'condition']

# Create a subgraph based on patient-related terms and their context
patient_subgraph = nx.Graph()

# Add patient-related terms to the subgraph
patient_subgraph.add_nodes_from(patient_terms)

# Iterate over the co-occurrence network and add relevant edges to the subgraph
for word1, word2 in cooc_network.edges():
    if word1 in patient_terms and word2 in top_keywords:
        patient_subgraph.add_edge(word1, word2, weight=cooc_network[word1][word2]['weight'])

# Perform community detection using Louvain algorithm
partition = community.best_partition(cooc_network)

# Create a dictionary to store the nodes for each cluster
cluster_nodes = {}
for node, cluster_id in partition.items():
    if cluster_id not in cluster_nodes:
        cluster_nodes[cluster_id] = [node]
    else:
        cluster_nodes[cluster_id].append(node)

# Create subgraphs for each cluster
cluster_subgraphs = []
for nodes in cluster_nodes.values():
    subgraph = cooc_network.subgraph(nodes).copy()
    cluster_subgraphs.append(subgraph)

# Visualize each cluster subgraph
for i, subgraph in enumerate(cluster_subgraphs):
    plt.figure(figsize=(10, 10))
    pos = nx.kamada_kawai_layout(subgraph)
    nx.draw_networkx(subgraph, pos=pos, with_labels=True, node_size=800, node_color='lightblue',
                     font_size=12, font_weight='bold', edge_color='gray', alpha=0.7, width=1.5)
    plt.title(f"Cluster {i+1} Subgraph")
    plt.axis('off')
    plt.show()
