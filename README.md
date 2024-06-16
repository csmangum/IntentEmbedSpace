# IntentEmbedSpace

Proof-of-concept of an intention embedding space to classify an incoming message with an intention without training a model.

Uses the embedding layer of a pre-trained lightweight tranformer to encode sentence context into vector space where new messages are encoded then use k-nearest neighbors to infer intent.
