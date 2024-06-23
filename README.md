# IntentEmbedSpace

Proof-of-concept of an intention embedding space to classify an incoming message with an intention without training a model.

Uses the embedding layer of a pre-trained lightweight transformer to encode sentence context into vector space where new messages are encoded then use k-nearest neighbors to infer intent from a pre-calculated intention space.

![Intent Embedding Space](docs/pca_clusters.png)

## Usage

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

### To run an interactive web app

```bash
python app.py
```

![Web App](docs/web-app.png)
