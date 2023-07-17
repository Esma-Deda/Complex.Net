import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer  
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
import PyPDF4
from fuzzywuzzy import fuzz
from nltk.corpus import wordnet


# Function to process the text and extract keywords related to clinician-patient communication
def process_clinician_patient_keywords(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Removing Stop Words and unwanted tokens
    stop_words = set(stopwords.words('english'))
    excluded_words = ['tx', 'butowpn', 'md', 'streetjrrl', 'tion', 'inturn', 'tion', 'thus', 'patient', 'doctor', 'physician']
    
    # Lemmatize the keywords
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words and token.isalpha() and len(token) > 1 and token.lower() not in excluded_words]

    # Filter keywords based on semantic similarity to relevant terms
    relevant_terms = ['patient', 'doctor', 'physician']
    filtered_tokens = []
    for token in lemmatized_tokens:
        synsets_token = wordnet.synsets(token)
        if any(synsets_token):
            for term in relevant_terms:
                synsets_term = wordnet.synsets(term)
                if any(synsets_term):
                    max_similarity = max(synsets_token[0].path_similarity(synset_term) for synset_term in synsets_term)
                    if max_similarity is not None and max_similarity >= 0.2:  # You can adjust the similarity threshold as needed
                        filtered_tokens.append(token)
                        break

    return filtered_tokens



# Function to create co-occurrence network
def create_cooccurrence_network(keywords):
    # Create a directed co-occurrence network (digraph)
    cooc_network = nx.DiGraph()

    # Add nodes to the network with 'label' attributes
    for keyword in keywords:
        cooc_network.add_node(keyword, label=keyword)

    # Compute co-occurrence frequency and add directed edges to the network
    for word1, word2 in combinations(keywords, 2):
        if cooc_network.has_edge(word1, word2):
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

# Process the article text and extract keywords related to clinician-patient communication
clinician_patient_keywords = process_clinician_patient_keywords(article_text)

# Calculate keyword frequencies
keyword_counts = Counter(clinician_patient_keywords)

# Get the 40 most important keywords based on their frequency
num_keywords = 40
top_keywords = [keyword for keyword, count in keyword_counts.most_common(num_keywords)]

# Create the co-occurrence network
cooc_network = create_cooccurrence_network(top_keywords)

# Visualize the directed clinician-patient communication-related subgraph
plt.figure(figsize=(12, 6))
pos = nx.spring_layout(cooc_network, seed=42)
nx.draw(cooc_network, pos=pos, with_labels=True, node_size=800, node_color='lightblue',
        font_size=12, font_weight='bold', edge_color='gray', alpha=0.7, width=1.5, arrows=True)
plt.title('Directed Co-occurrence Network of Clinician-Patient Communication Keywords')
plt.show()

# Function to calculate the average weight of edges in the co-occurrence network
def calculate_average_edge_weight(graph):
    total_weight = 0
    total_edges = graph.number_of_edges()

    for _, _, weight in graph.edges.data('weight'):
        total_weight += weight

    return total_weight / total_edges

# Calculate the average edge weight of the co-occurrence network
average_edge_weight = calculate_average_edge_weight(cooc_network)
print(f"Average Edge Weight: {average_edge_weight:.2f}")
