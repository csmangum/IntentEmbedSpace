import json

import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load pre-trained Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load labeled phrases from data.json
with open("data.json", "r") as file:
    data = json.load(file)
labeled_phrases = [(data[i]["text"], data[i]["intention"]) for i in range(len(data))]

# Separate phrases and labels
phrases, intentions = zip(*labeled_phrases)
# Encode phrases
phrase_embeddings = model.encode(phrases)

# Apply PCA to reduce to 2 dimensions or 3 dimensions
pca_2d = PCA(n_components=2)
reduced_embeddings_2d = pca_2d.fit_transform(phrase_embeddings)

pca_3d = PCA(n_components=3)
reduced_embeddings_3d = pca_3d.fit_transform(phrase_embeddings)

# Create a color map for different intentions
intention_set = set(intentions)
intention_colors = {
    intention: px.colors.qualitative.Plotly[i]
    for i, intention in enumerate(intention_set)
}

# Nearest Neighbors to classify intent
nbrs = NearestNeighbors(n_neighbors=5, metric="cosine").fit(phrase_embeddings)


def create_plot(sample_input, sample_reduced_embedding, majority_intent, dimensions):
    if dimensions == "2d":
        fig = px.scatter(
            x=reduced_embeddings_2d[:, 0],
            y=reduced_embeddings_2d[:, 1],
            color=intentions,
            color_discrete_map=intention_colors,
            title="PCA of Phrase Embeddings (2D)",
        )
        fig.add_trace(
            go.Scatter(
                x=[sample_reduced_embedding[0][0]],
                y=[sample_reduced_embedding[0][1]],
                mode="markers",
                marker=dict(color="black", size=15, symbol="x"),
                name=f"Input: {sample_input} (Intent: {majority_intent})",
            )
        )
    else:
        fig = px.scatter_3d(
            x=reduced_embeddings_3d[:, 0],
            y=reduced_embeddings_3d[:, 1],
            z=reduced_embeddings_3d[:, 2],
            color=intentions,
            color_discrete_map=intention_colors,
            title="PCA of Phrase Embeddings (3D)",
        )
        fig.add_trace(
            go.Scatter3d(
                x=[sample_reduced_embedding[0][0]],
                y=[sample_reduced_embedding[0][1]],
                z=[sample_reduced_embedding[0][2]],
                mode="markers",
                marker=dict(color="black", size=15, symbol="x"),
                name=f"Input: {sample_input} (Intent: {majority_intent})",
            )
        )
    return fig


@app.route("/", methods=["GET", "POST"])
def index():
    sample_input = ""
    majority_intent = ""
    selected_method = "nearest_neighbors"
    selected_dimensions = "2d"
    fig = create_plot("", [[0, 0]], "", selected_dimensions)

    if request.method == "POST":
        sample_input = request.form["sentence"]
        selected_method = request.form.get("method", "nearest_neighbors")
        selected_dimensions = request.form.get("dimensions", "2d")

        sample_embedding = model.encode([sample_input])
        if selected_dimensions == "2d":
            sample_reduced_embedding = pca_2d.transform(sample_embedding)
        else:
            sample_reduced_embedding = pca_3d.transform(sample_embedding)

        if selected_method == "nearest_neighbors":
            distances, indices = nbrs.kneighbors(sample_embedding)
            neighbor_intents = [intentions[i] for i in indices[0]]
            majority_intent = max(set(neighbor_intents), key=neighbor_intents.count)
        elif selected_method == "fuzzy_search":
            best_match = process.extractOne(sample_input, phrases)
            majority_intent = next(
                intention
                for phrase, intention in labeled_phrases
                if phrase == best_match[0]
            )

        fig = create_plot(
            sample_input, sample_reduced_embedding, majority_intent, selected_dimensions
        )

    graph_html = fig.to_html(full_html=False)
    return render_template(
        "index.html",
        graph_html=graph_html,
        sample_input=sample_input,
        majority_intent=majority_intent,
        selected_method=selected_method,
        selected_dimensions=selected_dimensions,
    )


if __name__ == "__main__":
    app.run(debug=True)
