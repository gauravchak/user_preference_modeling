# user_preference_modeling

Multiple ways to model user preference in recommender systems

Modeling the preference of the user as an input to the retrieval or ranking model has been a successful strategy in recommender systems. 

The conventional approach is using an embedding table lookup.
![user_id_embedding_lookup](./images/user_id_embedding_lookup.png)

We will also look at the schematic of an implementation using Deep Hash Embeddings.
![deep_hash_embeddings](./images/deep_hash_embeddings.png)

Then we will look at an approach where we reuse the machinery of Deep Hash Embeddings but seed it with an embedding that is looked up in a relatively small table as a function of the user's features (not including user id)
![user_feature_based_lookup](./images/user_feature_based_lookup.png)

## Testing

Run `pytest tests/*` from main directory.