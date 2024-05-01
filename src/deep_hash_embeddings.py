from typing import List

import torch
import torch.nn as nn

from src.multi_task_estimator import MultiTaskEstimator


class DHERepresentation(MultiTaskEstimator):
    """ Same as MultiTaskEstimator except using Deep hash Embeddings idea """

    def __init__(
        self,
        num_tasks: int,
        user_id_embedding_dim: int,
        user_features_size: int,
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        item_features_size: int,
        cross_features_size: int,
        user_value_weights: List[float],
        dhe_stack_in_embedding_dim: int,
    ) -> None:
        """
        params:
            num_tasks (T): The tasks to compute estimates of
            user_id_embedding_dim (DU): internal dimension
            user_features_size (IU): input feature size for users
            item_id_hash_size: the size of the embedding table for items
            item_id_embedding_dim (DI): internal dimension
            item_features_size: (II) input feature size for items
            cross_features_size: (IC) size of cross features
            user_value_weights: T dimensional weights, such that a linear
            combination of point-wise immediate rewards is the best predictor
            of long term user satisfaction.
            dhe_stack_in_embedding_dim (D_dhe_in): input emb dim for DHE
        """
        super(DHERepresentation, self).__init__(
            num_tasks=num_tasks,
            user_id_embedding_dim=user_id_embedding_dim,
            user_features_size=user_features_size,
            item_id_hash_size=item_id_hash_size,
            item_id_embedding_dim=item_id_embedding_dim,
            item_features_size=item_features_size,
            cross_features_size=cross_features_size,
            user_value_weights=user_value_weights,
        )
        # In DHE paper this was more than DU
        # In Twitter, GNN they found something similar to work where
        # the final layer was about one third of the start input dim.
        self.dhe_stack_in: int = dhe_stack_in_embedding_dim
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
