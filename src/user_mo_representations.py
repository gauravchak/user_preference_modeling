from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.user_cohort_embeddings import UserCohortRepresentation


class UserMORepresentations(UserCohortRepresentation):
    """
    This is the kicthen soup idea that has proliferated in ML recently.
    Here we use both 
    - the table lookup approach of how user id embeddings
    are implemented in UseridLookupRepresentation.
    - the cohort/cluster embeddings of UserCohortRepresentation
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
        user_id_embedding_dim: int,
        user_features_size: int,
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        item_features_size: int,
        cross_features_size: int,
        user_value_weights: List[float],
        cohort_table_size: int,
        cohort_lookup_dropout_rate: float,
        cohort_enable_topk_regularization: bool,
        user_id_hash_size: int,
    ) -> None:
        super(UserMORepresentations, self).__init__(
            num_tasks=num_tasks,
            user_id_embedding_dim=user_id_embedding_dim,
            user_features_size=user_features_size,
            item_id_hash_size=item_id_hash_size,
            item_id_embedding_dim=item_id_embedding_dim,
            item_features_size=item_features_size,
            cross_features_size=cross_features_size,
            user_value_weights=user_value_weights,
            cohort_table_size=cohort_table_size,
            cohort_lookup_dropout_rate=cohort_lookup_dropout_rate,
            cohort_enable_topk_regularization=cohort_enable_topk_regularization,
        )
        # Setup id lookup
        self.user_id_embedding_arch = nn.Embedding(
            user_id_hash_size, user_id_embedding_dim)
        # Setup Mixture network
        self.mixture_layer = nn.Sequential(
            nn.Linear(user_features_size, 2),
            nn.ReLU()
        )

    def get_user_embedding(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
    ) -> torch.Tensor:
        user_id_lookup_embedding = self.user_id_embedding_arch(
            user_id
        )  # [B, DU]
        user_id_cohort_embedding = super().get_user_embedding(
            user_id=user_id,
            user_features=user_features
        )  # [B, DU]
        rep_wts = self.mixture_layer(user_features)
        rep_probs = F.softmax(rep_wts, dim=1)  # [B, 2]
        stacked_user_embeddings = torch.stack(
            [user_id_lookup_embedding, user_id_cohort_embedding], 
            dim=2
        )  # [B, DU, 2]
        user_id_embedding = torch.bmm(
            stacked_user_embeddings, rep_probs.unsqueeze(2)
        ).squeeze(-1)  # [B, DU]
        return user_id_embedding

