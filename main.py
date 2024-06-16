import json

import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Load pre-trained Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example labeled phrases with intentions
labeled_phrases = [
    ("Goodbye", "goodbye"),
    ("See you later", "goodbye"),
    ("Bye", "goodbye"),
    ("Farewell", "goodbye"),
    ("Take care", "goodbye"),
    ("Hello", "greeting"),
    ("Hi", "greeting"),
    ("Good morning", "greeting"),
    ("How are you?", "greeting"),
    ("What's up?", "greeting"),
]

# Load in data.json and format it into a list of tuples
with open("data.json", "r") as file:
    data = json.load(file)

labeled_phrases = [(data[i]["text"], data[i]["intention"]) for i in range(len(data))]

# Separate phrases and labels
phrases, intentions = zip(*labeled_phrases)
# Encode phrases
phrase_embeddings = model.encode(phrases)

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(phrase_embeddings)

# Create a color map for different intentions
intention_set = set(intentions)
intention_colors = {
    intention: "C" + str(i) for i, intention in enumerate(intention_set)
}

# Plot the reduced embeddings
plt.figure(figsize=(10, 6))
for embedding, intention in zip(reduced_embeddings, intentions):
    plt.scatter(
        embedding[0], embedding[1], color=intention_colors[intention], label=intention
    )

# Add a black dot for a sample input->embedding->PCA
sample_input = "See ya later"
sample_embedding = model.encode([sample_input])
sample_reduced_embedding = pca.transform(sample_embedding)
plt.scatter(
    sample_reduced_embedding[0][0],
    sample_reduced_embedding[0][1],
    color="black",
    marker="X",
)

# Create a legend to show the intention colors
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

# Label the plot
plt.title("PCA of Phrase Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Nearest Neighbors to classify intent of sample_input
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=5, metric="cosine").fit(phrase_embeddings)
distances, indices = nbrs.kneighbors(
    sample_embedding
)  # Removed the extra brackets around sample_embedding
print(f"Nearest neighbors for {sample_input}:")
for i in indices[0]:
    print(f"{phrases[i]}: {intentions[i]}")
