import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        sample_tokens, _ = self.forward_with_scores(logits, temperatures)
        return sample_tokens

    @torch.compile
    def forward_with_scores(self, logits: torch.Tensor, temperatures: torch.Tensor):
        scores = logits.float().div_(temperatures.unsqueeze(dim=1))
        noise = torch.empty_like(scores).exponential_(1).clamp_min_(1e-10).log_()
        scores.sub_(noise)
        sample_scores, sample_tokens = scores.max(dim=-1)
        return sample_tokens, sample_scores

    @torch.compile
    def greedy_with_scores(self, logits: torch.Tensor):
        sample_scores, sample_tokens = logits.float().max(dim=-1)
        return sample_tokens, sample_scores
