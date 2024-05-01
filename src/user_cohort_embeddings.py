from typing import List

import torch
import torch.nn as nn

from multi_task_estimator import MultiTaskEstimator


class UserCohortEstimator(MultiTaskEstimator):
    """
    Here to capture user preference, we infer the cluster/cohort this
    user is closest to based on user features and use the embeddings
    of that cluster.
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

