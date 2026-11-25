"""Pure-PyTorch GRPO core math, chunked token entropy, k3 KL estimator, and gradient-norm decomposition.

Implemented directly in PyTorch without TRL or vLLM dependencies so every loss term,
temperature scaling, gradient path, and numerical check is transparent and unit-tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class GRPOConfig:
    """Frozen hyperparameters for the MiniBooking hand-written GRPO trainer."""

    group_size: int = 4
    prompts_per_step: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    logprob_temperature: float = 0.7
    entropy_log_temperature: float = 1.0
    reg_temperature: float = 1.0
    adv_eps: float = 1e-8
    clip_eps: float = 0.2
    beta_kl: float = 0.04
    lr: float = 1e-5
    warmup_steps: int = 5
    max_grad_norm: float = 1.0
    ent_chunk: int = 32


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute group-relative advantages using population standard deviation (``unbiased=False``).

    Args:
        rewards: 1-D tensor of shape ``[N_prompts * group_size]`` or 2-D ``[N_prompts, group_size]``.
        group_size: Number of sampled completions ``K`` per prompt.
        eps: Numerical stability floor added to the denominator.

    Returns:
        ``(advantages_flat, group_mean, group_std)`` where ``group_mean`` and ``group_std``
        have shape ``[N_prompts]``.
    """
    r_mat = rewards.float().reshape(-1, group_size)
    mean = r_mat.mean(dim=1, keepdim=True)
    std = r_mat.std(dim=1, unbiased=False, keepdim=True)
    adv = ((r_mat - mean) / (std + eps)).reshape(-1)
    return adv, mean.squeeze(1), std.squeeze(1)


def token_logprobs_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    temperature: float = 0.7,
) -> torch.Tensor:
    """Gather temperature-scaled token log-probabilities ``log pi(a_t | s_t)`` in float32."""
    scaled = logits.float() / float(temperature)
    return scaled.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(scaled, dim=-1)


@torch.no_grad()
def masked_token_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    chunk: int = 32,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Chunked float32 full-vocabulary token entropy for monitoring (detached from autograd).

    Args:
        logits: Tensor of shape ``[B, L, V]``.
        mask: Binary tensor of shape ``[B, L]`` marking valid completion tokens.
        chunk: Sequence chunk size to bound peak memory on large vocabularies.
        temperature: Softmax temperature for entropy computation.

    Returns:
        Tensor of shape ``[B, L]`` with zero outside ``mask``.
    """
    bsz, seq_len, _ = logits.shape
    ent = torch.zeros(bsz, seq_len, dtype=torch.float32, device=logits.device)
    for start in range(0, seq_len, chunk):
        lp = torch.log_softmax(logits[:, start : start + chunk].detach().float() / float(temperature), dim=-1)
        ent[:, start : start + chunk] = -(lp.exp() * lp).sum(dim=-1)
    return ent * mask.to(ent.dtype)


def entropy_summary(ent: torch.Tensor, mask: torch.Tensor) -> Dict[str, Any]:
    """Summarize masked token entropies over active completion tokens."""
    values = ent[mask.bool()]
    if values.numel() == 0:
        return {"mean": None, "median": None, "n": 0}
    return {"mean": float(values.mean()), "median": float(values.float().median()), "n": int(values.numel())}


def seq_entropy_with_grad(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    *,
    chunk: int = 32,
) -> torch.Tensor:
    """Differentiable per-sequence mean full-vocabulary entropy ``[B]`` for loss regularization.

    Preserves the autograd graph so ``-lambda_entropy * H_reg`` contributes true analytical
    entropy gradients ``dH/dz_i = -(1/T) * p_i * (log p_i + H)`` to policy parameters.
    """
    _, seq_len, _ = logits.shape
    parts = []
    for start in range(0, seq_len, chunk):
        lp = torch.log_softmax(logits[:, start : start + chunk].float() / float(temperature), dim=-1)
        parts.append(-(lp.exp() * lp).sum(dim=-1))
    h_tok = torch.cat(parts, dim=1)
    m = mask.to(h_tok.dtype)
    return (h_tok * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def grpo_loss_components(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Compute per-sequence GRPO clipped surrogate loss and unbiased k3 KL divergence.

    k3 KL estimator (Schulman): ``exp(ref - lp) - (ref - lp) - 1 >= 0``.
    """
    ratio = torch.exp(logprobs - old_logprobs)
    adv_col = advantages[:, None]
    surr = torch.min(ratio * adv_col, torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_col)
    diff = ref_logprobs - logprobs
    kl = torch.exp(diff) - diff - 1.0
    m = mask.float()
    n_tok = m.sum(dim=1).clamp(min=1.0)
    seq_pg = (-surr * m).sum(dim=1) / n_tok
    seq_kl = (kl * m).sum(dim=1) / n_tok
    with torch.no_grad():
        stats = {
            "pg_sum": float(seq_pg.sum()),
            "kl_sum": float(seq_kl.sum()),
            "seq_kl": seq_kl.tolist(),
            "clip_frac_num": float((((ratio - 1.0).abs() > clip_eps).float() * m).sum()),
            "ntok": float(m.sum()),
        }
    return seq_pg, seq_kl, stats


def grad_l2_norm(grads: Sequence[Optional[torch.Tensor]]) -> float:
    """Compute global L2 norm over a sequence of optional gradient tensors."""
    return math.sqrt(sum(float((g.float() ** 2).sum()) for g in grads if g is not None))


def compute_grpo_objective(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    old_logprobs: Optional[torch.Tensor] = None,
    beta_kl: float = 0.04,
    lambda_entropy: float = 0.0,
    with_entropy_reg: bool = False,
    clip_eps: float = 0.2,
    logprob_temperature: float = 0.7,
    reg_temperature: float = 1.0,
    ent_chunk: int = 32,
    params_for_ratio: Optional[Sequence[torch.nn.Parameter]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute full single-batch GRPO loss with optional entropy regularization and gradient ratio.

    Loss formula:
        ``loss = mean(seq_pg) + beta_kl * mean(seq_kl) - lambda_entropy * mean(seq_H_reg)``
    """
    lam = float(lambda_entropy)
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError(f"lambda_entropy must be finite and >= 0, got {lam!r}")
    if not with_entropy_reg and lam != 0.0:
        raise ValueError("lambda_entropy must be 0.0 when with_entropy_reg=False")

    lp = token_logprobs_from_logits(logits, target_ids, temperature=logprob_temperature)
    old_lp = lp.detach() if old_logprobs is None else old_logprobs
    seq_pg, seq_kl, stats = grpo_loss_components(lp, old_lp, ref_logprobs, advantages, mask, clip_eps=clip_eps)

    pg_term = seq_pg.mean()
    kl_term = seq_kl.mean()
    need_h = with_entropy_reg or (params_for_ratio is not None)
    h_reg = seq_entropy_with_grad(logits, mask, temperature=reg_temperature, chunk=ent_chunk).mean() if need_h else None

    loss = pg_term + beta_kl * kl_term
    if with_entropy_reg:
        assert h_reg is not None and h_reg.requires_grad, "Entropy regularization term must carry autograd graph"
        loss = loss - lam * h_reg

    gn_pg = gn_h = ent_grad_ratio = None
    if params_for_ratio is not None:
        assert h_reg is not None
        g_pg = torch.autograd.grad(pg_term, params_for_ratio, retain_graph=True, allow_unused=True)
        g_h = torch.autograd.grad(h_reg, params_for_ratio, retain_graph=True, allow_unused=True)
        gn_pg = grad_l2_norm(g_pg)
        gn_h = grad_l2_norm(g_h)
        ent_grad_ratio = (lam * gn_h / gn_pg) if gn_pg > 0.0 else None

    info = {
        "loss": float(loss.detach()),
        "pg_loss": float(pg_term.detach()),
        "kl": float(kl_term.detach()),
        "clip_frac": stats["clip_frac_num"] / max(1.0, stats["ntok"]),
        "H_reg": float(h_reg.detach()) if h_reg is not None else None,
        "ent_loss_term": (lam * float(h_reg.detach())) if (with_entropy_reg and h_reg is not None) else 0.0,
        "gn_pg": gn_pg,
        "gn_H_reg": gn_h,
        "ent_grad_ratio": ent_grad_ratio,
        "seq_kl": stats["seq_kl"],
    }
    return loss, info
