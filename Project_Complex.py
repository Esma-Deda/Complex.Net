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
    excluded_words = ['tx','kuom', 'dc', 'stileswb', 'wardmm', 'butowpn', 'makoulg', 'wadmm', 'gafnia', 'kinmonthal','md', 'streetjrrl', 'tion', 'inturn', 'tion', 'thus', 'fujimorim', 'tattersallmh', 'brownrf', 'sabilityto', 'gordonhs']  
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


def filter_keywords_by_frequency(keywords, min_frequency):
    filtered_keywords = [keyword for keyword in keywords if keyword_counts[keyword] >= min_frequency]
    return filtered_keywords

min_frequency_threshold = 2

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

# Get the 40 most important keywords based on their frequency
num_keywords = 40
top_keywords = [keyword for keyword, count in keyword_counts.most_common(num_keywords)]

# Filter keywords based on the minimum frequency threshold
top_keywords = filter_keywords_by_frequency(top_keywords, min_frequency_threshold)

# Create the co-occurrence network
cooc_network = create_cooccurrence_network(top_keywords)

# Compute degree centrality
degree_centrality = nx.degree_centrality(cooc_network)

# Compute betweenness centrality
betweenness_centrality = nx.betweenness_centrality(cooc_network)

# Compute betweenness centrality
betweenness_centrality = nx.betweenness_centrality(cooc_network)

# Compute closeness centrality
closeness_centrality = nx.closeness_centrality(cooc_network)

# Compute PageRank
pagerank = nx.pagerank(cooc_network)

# Compute network density
network_density = nx.density(cooc_network)

# Compute degree assortativity
degree_assortativity = nx.degree_assortativity_coefficient(cooc_network)

# Compute the degree of each node
degrees = dict(cooc_network.degree())

# Compute the degrees of the neighboring nodes for each node
neighbor_degrees = []
for node in cooc_network.nodes():
    neighbor_degrees.append([degrees[neighbor] for neighbor in cooc_network.neighbors(node)])

# Compute degree correlation
degree_correlation = nx.degree_pearson_correlation_coefficient(cooc_network)

# Perform community detection using Louvain algorithm
partition = community.best_partition(cooc_network.to_undirected())

# Print the results
print("Degree Centrality:", degree_centrality)
print("Betweenness Centrality:", betweenness_centrality)
print("Closeness Centrality:", closeness_centrality)
print("PageRank:", pagerank)
print("Network Density:", network_density)
print("Degree Assortativity:", degree_assortativity)
print("Community Structure:", partition)
print("Degree Correlation:", degree_correlation)

# Calculate clustering coefficient
clustering_coefficient = nx.clustering(cooc_network)

# Print the clustering coefficient for each node
for node, coefficient in clustering_coefficient.items():
    print(f"Node {node}: Clustering Coefficient = {coefficient}")

# Draw the co-occurrence network
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(cooc_network)
nx.draw_networkx(cooc_network, pos=pos, with_labels=True, node_size=800, node_color='lightblue', font_size=12, font_weight='bold', edge_color='gray', alpha=0.7)
plt.axis('off')
plt.show()

# Plot the histogram of keyword frequencies
keyword_freq = [count for keyword, count in keyword_counts.most_common(num_keywords)]
plt.figure(figsize=(8, 6))
plt.hist(keyword_freq, bins=20, edgecolor='black')
plt.xlabel('Frequency')
plt.ylabel('Count')
plt.title('Histogram of Keyword Frequencies')
plt.show()
