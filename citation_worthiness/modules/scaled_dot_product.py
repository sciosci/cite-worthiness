import torch
from overrides import overrides
from allennlp.modules.attention.legacy_attention import Attention
import math

@Attention.register("scaled_dot_product")
class DotProductAttention(Attention):
    """
    Computes attention between a vector and a matrix using dot product.
    """
    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        sim_score = matrix.bmm(vector.unsqueeze(-1)).squeeze(-1)
        sim_score_scaled = sim_score / math.sqrt(sim_score.size(-1))
        return sim_score_scaled

