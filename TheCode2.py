import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx
from yake import KeywordExtractor
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    keywords = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]

    return ' '.join(keywords)

def extract_keywords(article_text):
    # Preprocess the text
    preprocessed_text = preprocess_text(article_text)

    # Extract keywords using YAKE
    kw_extractor = KeywordExtractor(lan="en", n=2, top=40, features=None)
    keywords = kw_extractor.extract_keywords(preprocessed_text)

    return [keyword[0] for keyword in keywords]

def compute_semantic_similarity(keywords):
    # Load pre-trained BERT model
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Create a network graph
    G = nx.Graph()
    for i in range(len(keywords)):
        G.add_node(keywords[i])

    # Calculate semantic similarity and add edges to the network
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            similarity_score = cosine_similarity([model.encode(keywords[i])], [model.encode(keywords[j])])[0][0]
            if similarity_score > 0.8:  # Adjust the threshold as needed
                G.add_edge(keywords[i], keywords[j], weight=similarity_score)

    return G

def visualize_network(G):
    plt.figure(figsize=(12, 10))

    # Adjust the layout algorithm to improve node spacing
    pos = nx.spring_layout(G, k=0.7, iterations=50, seed=30)

    # Customize node and edge sizes
    node_sizes = [50 * G.degree(node) for node in G.nodes()]
    edge_widths = [2.0 * G[u][v]['weight'] for u, v in G.edges()]

    # Draw the network with adjusted parameters
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', node_size=node_sizes, edge_color='gray',
                     width=edge_widths, font_size=10, font_weight='bold', alpha=0.8)

    plt.title('Keyword Semantic Similarity Network')
    plt.axis('off')
    plt.show()

# Example usage
file_path = '/Users/esmaisufi/Desktop/Articles/2.How-does-communication-heal--Pathways-linking-clinici_2009_Patient-Education.pdf'  # Replace with the correct file path

# Read the PDF article
with open(file_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    article_text = ""
    for page in pdf_reader.pages:
        article_text += page.extract_text()

# Extract keywords
keywords = extract_keywords(article_text)

# Compute semantic similarity and create the network
network = compute_semantic_similarity(keywords)

# Visualize the network
visualize_network(network)
