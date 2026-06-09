import torch
import torch.nn.functional as F
import numpy as np

class RicciAttentionMetrics:
    @staticmethod
    def entropy(attn_weights, eps=1e-8):
        attn = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + eps)
        entropy = -torch.sum(attn * torch.log(attn + eps), dim=-1)
        return entropy
    
    @staticmethod
    def temporal_singularity(attn_weights, eps=1e-8):
        entropy = RicciAttentionMetrics.entropy(attn_weights, eps)
        entropy_mean = entropy.mean(dim=0)
        max_entropy = np.log(attn_weights.shape[-1])
        entropy_norm = entropy_mean / max_entropy
        temporal_var = torch.var(entropy_norm, dim=-1, unbiased=False)
        sing_score = 1.0 - temporal_var
        return {
            'sing_score': sing_score,
            'entropy_sequence': entropy_norm,
            'temporal_variance': temporal_var
        }
    
    @staticmethod
    def cross_head_correlation(attn_weights):
        avg_attn = attn_weights.mean(dim=-1).mean(dim=-1)
        cov = torch.cov(avg_attn.T)
        std = torch.sqrt(torch.diag(cov)).unsqueeze(0)
        corr = cov / (std.T @ std + 1e-8)
        mean_corr = (corr.sum(dim=-1) - 1) / (corr.shape[-1] - 1)
        return mean_corr

    @staticmethod
    def combined_singularity(attn_weights):
        temporal = RicciAttentionMetrics.temporal_singularity(attn_weights)['sing_score']
        cross = RicciAttentionMetrics.cross_head_correlation(attn_weights)
        combined = temporal * torch.sigmoid(-cross * 5)
        return combined
