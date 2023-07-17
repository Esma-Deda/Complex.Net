import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import PyPDF4
import community
from collections import Counter

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

# Function to create co-occurrence network based on average distance
def create_cooccurrence_network_with_distance(keywords):
    # Create a co-occurrence network
    cooc_network = nx.Graph()

    # Add nodes to the network
    cooc_network.add_nodes_from(keywords)

    # Compute the shortest path lengths between all pairs of nodes
    shortest_path_lengths = dict(nx.shortest_path_length(cooc_network))

    # Compute the inverse of the shortest path lengths and set edge weights accordingly
    for word1, word2 in combinations(keywords, 2):
        if word1 not in shortest_path_lengths or word2 not in shortest_path_lengths[word1]:
            cooc_network.add_edge(word1, word2, weight=0)  # Set weight to 0 for invalid distances
        else:
            distance = shortest_path_lengths[word1][word2]
            if distance == 0:
                cooc_network.add_edge(word1, word2, weight=0)  # Set weight to 0 for disconnected nodes
            else:
                weight = 1 / distance
                cooc_network.add_edge(word1, word2, weight=weight)

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

# Calculate keyword frequencies
keyword_counts = Counter(keywords)

# Get the 40 most important keywords based on their frequency
num_keywords = 40
top_keywords = [keyword for keyword, count in keyword_counts.most_common(num_keywords)]

# Create the co-occurrence network with average distance as edge weights
cooc_network = create_cooccurrence_network_with_distance(top_keywords)

# ... (The rest of the code remains the same)

# Draw the patient context subgraph
plt.figure(figsize=(12, 6))
pos = nx.spring_layout(cooc_network, k=0.7)
nx.draw_networkx(cooc_network, pos=pos, with_labels=False, node_size=800, node_color='lightblue',
                 font_size=12, font_weight='bold', edge_color='gray', alpha=0.7, width=1.5)
nx.draw_networkx_labels(cooc_network, pos=pos, font_size=10, font_weight='normal')
plt.axis('off')
plt.title('Network graph')
plt.show()

# Calculate keyword frequencies
keyword_freq = [count for keyword, count in keyword_counts.most_common(num_keywords)]

# Plot the histogram of keyword frequencies
plt.figure(figsize=(12, 6))
plt.hist(keyword_freq, bins=20, edgecolor='black')
plt.xlabel('Frequency')
plt.ylabel('Count')
plt.title('Histogram of Keyword Frequencies')
plt.show()
