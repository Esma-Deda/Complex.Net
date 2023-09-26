# Importing all the neccesay libraries
import networkx as nx
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import PyPDF2
import csv
from textblob import TextBlob
from textstat import flesch_reading_ease
from wordcloud import WordCloud
from collections import Counter
from networkx.algorithms.community import girvan_newman

# Function to extract text from the PDF file
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Extract text from the PDF file
file_path = '/Users/esmaisufi/Desktop/Articles/4.The-medical-consultation-viewed-as-a-value-chain--A-neuro_2009_Patient-Educa.pdf'
pdf_text = extract_text_from_pdf(file_path)

# Preprocessing function
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    words = word_tokenize(text.lower())
    words = [ps.stem(w) for w in words if w.isalnum() and w not in stop_words and not w.isdigit()]
    return words

# Tokenize and preprocess the entire text
all_words = preprocess_text(pdf_text)


# Additional Analysis: 
# Word Frequency and Word Cloud
word_freq = Counter(all_words)
wordcloud = WordCloud(width=800, height=400).generate(pdf_text)
output_file = 'word_frequency.csv'

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Word', 'Frequency'])
    for word, freq in word_freq.items():
        writer.writerow([word, freq])
# Sentiment Analysis
sentiment = TextBlob(pdf_text)
sentiment_score = sentiment.sentiment.polarity
print("Sentiment Score:", sentiment_score)
# Readability Analysis
readability_score = flesch_reading_ease(pdf_text)
print("Readability Score:", readability_score)


# Create a set of unique words
all_words = preprocess_text(pdf_text)

# Count the number of words after preprocessing
num_words_after_preprocessing = len(all_words)
print("Number of words after preprocessing:", num_words_after_preprocessing)

# Create a directed graph
G = nx.DiGraph()

# Calculate average distance between each word pair
average_distance = {}  # Dictionary to store average distances

# Calculate the average distance for each word pair
for i, word1 in enumerate(all_words):
    for j in range(i + 1, min(i + 20, len(all_words))):  # Limited the search to the next 20 words for efficiency
        word2 = all_words[j]
        if word1 != word2:
            if (word1, word2) not in average_distance:
                # Initialize distance to 0
                average_distance[(word1, word2)] = 0

            # Calculating the distance and adding it to the average
            average_distance[(word1, word2)] += (j - i)

# Create weighted edges based on the inverse of average distances
for word_pair, total_distance in average_distance.items():
    word1, word2 = word_pair
    count = all_words.count(word1)  # Number of times word1 occurs
    if count > 0:
        average_distance[word_pair] = total_distance / count
        weight = 1 / (average_distance[word_pair] + 1)  # Apply the inverse function
        G.add_edge(word1, word2, weight=weight)


# Set a threshold as a multiple of the average weight
threshold_multiplier = 2.59
threshold = sum(edge[2]['weight'] for edge in G.edges(data=True)) / G.number_of_edges() * threshold_multiplier

# Create a new graph to store filtered edges
filtered_G = nx.DiGraph()

# Iterate through the edges of the original graph and add them to the filtered graph if they meet the threshold
for edge in G.edges(data=True):
    if edge[2]['weight'] >= threshold:
        filtered_G.add_edge(edge[0], edge[1], weight=edge[2]['weight'])

# Analyze the total link weight distribution above the threshold.
edge_weights_above_threshold = [edge[2]['weight'] for edge in filtered_G.edges(data=True) if edge[2]['weight'] >= threshold]

# Plot a histogram of the edge weights
plt.hist(edge_weights_above_threshold, bins=50, alpha=0.5, color='b', edgecolor='black')
plt.xlabel('Edge Weight')
plt.ylabel('Frequency')
plt.title('Total Link Weight Distribution')
plt.grid(True)
plt.yscale('log')
plt.show()

# Calculate the total number of connections in the graph
total_connections = len(filtered_G.edges())
print(f"Total Number of Connections: {total_connections}")

# Visualize the network with the circular layout
pos = nx.random_layout(filtered_G)
plt.figure(figsize=(8, 8))
nx.draw(filtered_G, pos, with_labels=True, node_size=50, font_size=8, font_color='black', node_color='skyblue', edge_color='gray', width=0.5, arrows=False)
plt.title('Filtered Word Network (Spring Layout with More Iterations)')
plt.show()


# Additional Network Analysis

# 1. Centrality Measures
degree_centrality = nx.degree_centrality(filtered_G)
betweenness_centrality = nx.betweenness_centrality(filtered_G)
closeness_centrality = nx.closeness_centrality(filtered_G)

# Save degree centrality to a CSV file
with open('degree_centrality.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Node', 'Degree Centrality'])
    for node, centrality in degree_centrality.items():
        writer.writerow([node, centrality])


# Save betweenness centrality to a CSV file
with open('betweenness_centrality.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Node', 'Betweenness Centrality'])
    for node, centrality in betweenness_centrality.items():
        writer.writerow([node, centrality])

 # Save closeness centrality to a CSV file
with open('closeness_centrality.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Node', 'Closeness Centrality'])
    for node, centrality in closeness_centrality.items():
        writer.writerow([node, centrality])
       
# 2. PageRank Analysis (for directed graphs)
pagerank = nx.pagerank(filtered_G)
output_file = 'pagerank_values.csv'

# Save PageRank values to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Node', 'PageRank'])
    for node, value in pagerank.items():
        writer.writerow([node, value])

# Calculate the overall PageRank value for the entire network
overall_pagerank = sum(pagerank.values())
print("Overall PageRank Value:", overall_pagerank)

# 3. Network Density
network_density = nx.density(filtered_G)
print("Network Density:", network_density)

# 4. Degree Assortativity
degree_assortativity = nx.degree_assortativity_coefficient(filtered_G)
print("Degree Assortativity:", degree_assortativity)

# 5. Degree Correlation
degree_correlation = nx.degree_pearson_correlation_coefficient(filtered_G)
print("Degree Correlation:", degree_correlation)

# 6. Clustering coefficient
clustering_coefficient = nx.average_clustering(filtered_G)
print("Clustering Coefficient:", clustering_coefficient)

# 7. Community Detection

# Convert the directed graph to an undirected graph
undirected_G = filtered_G.to_undirected()
communities = girvan_newman(undirected_G)

# Convert communities to a list for easier processing
community_list = [list(community) for community in next(communities)]

# Create a mapping of nodes to communities
community_mapping = {}
for i, community in enumerate(community_list, start=1):
    for node in community:
        community_mapping[node] = i

# Save community detection results to a CSV file
output_file = 'community_detection_results.csv'
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Node', 'Community'])
    for node, community_id in community_mapping.items():
        writer.writerow([node, community_id])

# 8. Connected components
connected_components = list(nx.weakly_connected_components(filtered_G))
output_file = 'connected_components.csv'

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Component', 'Nodes'])
    for i, component in enumerate(connected_components, start=1):
        writer.writerow([f'Component {i}', ', '.join(component)])

