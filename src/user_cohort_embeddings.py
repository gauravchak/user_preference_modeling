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
        cohort_table_size: int,
        cohort_lookup_dropout_rate: float=0.5,
        cohort_enable_topk_regularization: bool=False,
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
        # Initialize the cohort embedding matrix with random values
        self.cohort_embedding_matrix = nn.Parameter(
            torch.randn(cohort_table_size, self.dhe_stack_in)
        )  # [cohort_table_size, self.dhe_stack_in]
        self.cohort_enable_topk_regularization: bool = (
            cohort_enable_topk_regularization
        )
        self.topk:int = 1
        # Get cohort addressing from user features.
        # Input: [B, user features]
        # Output: [B, cohort_table_size]
        self.cohort_addressing_layer = nn.Sequential(
            nn.Linear(
                in_features=user_features_size,
                out_features=cohort_table_size),
            # Adding dropout could increase generalization
            # by allowing multiple clusters in the table to learn
            # from the behavior of a user.
            nn.Dropout(p=cohort_lookup_dropout_rate)
        )

    def get_user_embedding(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
    ) -> torch.Tensor:
        """
        Returns: [B, user_id_embedding_dim]
        """
        # Pass user features through the cohort addressing layer
        cohort_affinity = self.cohort_addressing_layer(user_features)  # [B, H]

        if self.cohort_enable_topk_regularization:
            # Apply top-k=1 to get the indices of the top k values
            # This ensures that the sum along dim 1 will finally be 1
            _, topk_indices = torch.topk(cohort_affinity, k=self.topk, dim=1)

            # Change cohort_affinity to a binary tensor initialized to 0
            cohort_affinity.fill_(0)

            # Set the selected indices to 1/self.topk
            # Note that if you have topk > 1 then this ensures that
            # the sum of values in dim=1 is still 1
            cohort_affinity.scatter_(
                dim=1, index=topk_indices,
                value=(1/self.topk)
            )

        # Perform matrix multiplication with the embedding matrix
        cohort_embedding_in = torch.matmul(
            cohort_affinity, self.cohort_embedding_matrix
        )  # [B, H] * [H, dhe_in] -> [B, dhe_in]
        user_id_embeddings = self.dhe_stack(cohort_embedding_in)  # [B, DU]
        return user_id_embeddings
