# Multiple ways to model user preference in recommender systems

Modeling the preference of the user as an input to the retrieval or ranking model has been a successful strategy in recommender systems. In this repo, we will show how to do it effectively. We will use the example of a ranking model but the approach equally applies to a retrieval model. The overall schematic of a ranking model is in ![Fig 0: schematic_multi_task_estimator](./images/schematic_multi_task_estimator.png)

The conventional approach of representing a user is using an embedding table lookup as shown in the image below and implemented in [multi_task_estimator.py](./src/multi_task_estimator.py).
![Fig 1: user_id_embedding_lookup](./images/user_id_embedding_lookup.png)


We will also look at the schematic of an implementation using Deep Hash Embeddings.
![Fig 2: deep_hash_embeddings](./images/deep_hash_embeddings.png)

Then we will look at an approach where we reuse the machinery of Deep Hash Embeddings but seed it with an embedding that is looked up in a relatively small table as a function of the user's features (not including user id)
![Fig 3: user_feature_based_lookup](./images/user_feature_based_lookup.png)

Finally we will put id lookup and cohort lookup together using and idea from [this paper from Google](https://arxiv.org/abs/2210.14309). This image from the paper captures the idea:
![Fig 4: Memorization vs Generalization](./images/memorization_vs_generalization.png)

The implementation in our repository is:
![Fig 5: Mixture of Representations](./images/moe_cohort_plus_id_lookup.png)

### Customization
If you want to allocate more of your memorization capacity to a certain cohort, for instance you could care more about US users, you could do that by encoding the weight in the loss function and perhaps adding the country / feature in the input to the Mixture of Representations tower.

## Contributing

Run `pytest tests/*` from main directory before submitting a PR.
