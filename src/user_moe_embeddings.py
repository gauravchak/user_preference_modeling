from typing import List

import torch
import torch.nn as nn

from multi_task_estimator import MultiTaskEstimator


class UserCohortEstimator(MultiTaskEstimator):
    """
    This is the kicthen soup idea that has proliferated in ML recently.
    Here we use both 
    - the table lookup approach of how user id embeddings
    are implemented in MultiTaskEstimator.
    - the cohort/cluster embeddings of UserCohortEstimator
    Then we mix it up using a [(k * emb_lookup) + ((1 - k) * emb_cohort)]
    approach that is popularly known as Mixture of Experts, where k is
    a function of user features.
    If you are implementing this, I suggest you to verify that k is higher
    for power users and lower for cold start users/marginal users. You can
    see an example of this in paper https://arxiv.org/abs/2210.14309.
    """

    def __init__(
        self,
        num_tasks: int,
        user_id_hash_size: int,
        user_id_embedding_dim: int,
        user_features_size: int,
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        item_features_size: int,
        cross_features_size: int,
        user_value_weights: List[float],
    ) -> None:
        super(UserCohortEstimator, self).__init__(
            num_tasks,
            user_id_hash_size,
            user_id_embedding_dim,
            user_features_size,
            item_id_hash_size,
            item_id_embedding_dim,
            item_features_size,
            cross_features_size,
            user_value_weights,
        )

