from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

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
    ("What's up?", "greeting")
]

# Separate phrases and labels
phrases, intentions = zip(*labeled_phrases)
# Encode phrases
phrase_embeddings = model.encode(phrases)

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(phrase_embeddings)

# Create a color map for different intentions
intention_colors = {
    "goodbye": "red",
    "greeting": "blue"
}

# Plot the reduced embeddings
plt.figure(figsize=(10, 6))
for embedding, intention in zip(reduced_embeddings, intentions):
    plt.scatter(embedding[0], embedding[1], color=intention_colors[intention], label=intention)

# Create a legend to show the intention colors
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

# Label the plot
plt.title("PCA of Phrase Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
