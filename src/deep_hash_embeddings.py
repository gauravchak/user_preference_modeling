from typing import List

import torch
import torch.nn as nn

from multi_task_estimator import MultiTaskEstimator


class DHEEstimator(MultiTaskEstimator):
    """ Same as MultiTaskEstimator except using Deep hash Embeddings idea """

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
        super(DHEEstimator, self).__init__(
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
        self.dhe_stack = nn.Sequential(
            nn.Linear(item_id_embedding_dim, item_id_embedding_dim),
            nn.ReLU(),
            nn.Linear(item_id_embedding_dim, item_id_embedding_dim),
            nn.ReLU(),
            nn.Linear(item_id_embedding_dim, item_id_embedding_dim),
            nn.ReLU(),
            nn.Linear(item_id_embedding_dim, item_id_embedding_dim)
        )

    def hash_fn(
        self,
        user_id: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """WIP, Need to replace with a proper hash function"""
        return torch.randn(user_id.shape[0], self.item_id_embedding_dim)

    def get_user_embedding(
        self, 
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
    ) -> torch.Tensor:
        user_hash = self.hash_fn(user_id)  # [B, D]
        user_id_embeddings = self.dhe_stack(user_hash)
        return user_id_embeddings
