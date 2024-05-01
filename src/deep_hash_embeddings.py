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
        # In DHE paper this was more than DU
        # In Twitter, GNN they found something similar to work where
        # the final layer was about one third of the start input dim.
        self.dhe_stack_in: int = user_id_embedding_dim
        self.dhe_stack = nn.Sequential(
            nn.Linear(self.dhe_stack_in, user_id_embedding_dim),
            nn.ReLU(),
            nn.Linear(user_id_embedding_dim, user_id_embedding_dim),
            nn.ReLU(),
            nn.Linear(user_id_embedding_dim, user_id_embedding_dim),
            nn.ReLU(),
            nn.Linear(user_id_embedding_dim, user_id_embedding_dim)
        )

    def hash_fn(
        self,
        user_id: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Returns [B, self.dhe_stack_in]
        WIP, Need to replace with a proper hash function
        """
        return torch.randn(user_id.shape[0], self.dhe_stack_in)  # [B, D_dhe_in]

    def get_user_embedding(
        self, 
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
    ) -> torch.Tensor:
        """
        Returns: [B, user_id_embedding_dim]
        """
        user_hash = self.hash_fn(user_id)  # [B, D_dhe_in]
        user_id_embeddings = self.dhe_stack(user_hash)  # [B, DU]
        return user_id_embeddings
