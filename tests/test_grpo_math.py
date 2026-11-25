"""Unit tests for pure-PyTorch GRPO loss, k3 KL estimator, group advantages, and analytical entropy gradients."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from minibooking_rlvr.rl.grpo import (  # noqa: E402
    compute_group_advantages,
    compute_grpo_objective,
    entropy_summary,
    grpo_loss_components,
    masked_token_entropy,
    seq_entropy_with_grad,
)


def test_group_advantages_zero_variance_and_normalization():
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    adv, mean, std = compute_group_advantages(rewards, group_size=4, eps=1e-8)
    assert torch.allclose(adv[:4], torch.zeros(4), atol=1e-6)
    assert mean[1].item() == pytest.approx(0.25)
    assert std[1].item() == pytest.approx(math.sqrt(0.1875))
    assert adv[4:].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert adv[4:].std(unbiased=False).item() == pytest.approx(1.0, abs=1e-5)


def test_masked_and_differentiable_entropy_exact_and_analytical_gradient():
    torch.manual_seed(0)
    vocab = 50
    logits = torch.randn(2, 7, vocab) * 3.0
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0]])

    e_chunk = masked_token_entropy(logits, mask, chunk=3, temperature=1.0)
    e_full = masked_token_entropy(logits, mask, chunk=1000, temperature=1.0)
    e_ref = -(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1) * mask
    assert torch.allclose(e_chunk, e_full, atol=1e-6)
    assert torch.allclose(e_chunk, e_ref, atol=1e-5)
    assert torch.all(e_chunk[mask == 0] == 0)

    # Monitoring version must not retain autograd graph
    assert masked_token_entropy(logits.clone().requires_grad_(True), mask).grad_fn is None

    # Uniform distribution achieves log(V)
    u = masked_token_entropy(torch.zeros(1, 1, vocab), torch.ones(1, 1))
    assert float(u) == pytest.approx(math.log(vocab), abs=1e-5)

    # Summary matches active tokens
    summ = entropy_summary(e_chunk, mask)
    assert summ["n"] == 8 and summ["mean"] > 0

    # Differentiable entropy matches monitoring mean and exact analytical gradient dH/dz = -(1/T)*p*(log p + H)
    for temp in (1.0, 0.7):
        z = logits.clone().float().requires_grad_(True)
        h_seq = seq_entropy_with_grad(z, mask.float(), temperature=temp, chunk=3)
        mon = masked_token_entropy(logits, mask, temperature=temp)
        assert torch.allclose(h_seq, mon.sum(1) / mask.sum(1), atol=1e-5)

        zz = logits.clone().float().requires_grad_(True)
        lp = torch.log_softmax(zz / temp, dim=-1)
        h_tok = -(lp.exp() * lp).sum(dim=-1)
        (g_auto,) = torch.autograd.grad(h_tok.sum(), zz)
        with torch.no_grad():
            lp0 = torch.log_softmax(logits.float() / temp, dim=-1)
            p0 = lp0.exp()
            h0 = -(p0 * lp0).sum(dim=-1, keepdim=True)
            g_ana = -(1.0 / temp) * p0 * (lp0 + h0)
        assert torch.allclose(g_auto, g_ana, atol=1e-5)

    # Stepping along -grad(-lambda * H) increases entropy
    z = logits.clone().float().requires_grad_(True)
    loss_ent = -0.01 * seq_entropy_with_grad(z, mask.float(), temperature=1.0).sum()
    (grad_z,) = torch.autograd.grad(loss_ent, z)
    with torch.no_grad():
        h_before = float(seq_entropy_with_grad(logits.float(), mask.float(), 1.0).sum())
        h_after = float(seq_entropy_with_grad(logits.float() - 50.0 * grad_z, mask.float(), 1.0).sum())
    assert h_after > h_before


def test_k3_kl_nonnegative_and_grpo_objective_gradient_ratio():
    torch.manual_seed(7)
    bsz, seq_len, vocab = 4, 6, 32
    head = torch.nn.Linear(8, vocab)
    feats = torch.randn(bsz, seq_len, 8)
    target_ids = torch.randint(0, vocab, (bsz, seq_len))
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])

    logits = head(feats)
    ref_lp = torch.log_softmax(logits.detach() * 0.9, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    seq_pg, seq_kl, stats = grpo_loss_components(
        torch.log_softmax(logits / 0.7, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1),
        torch.log_softmax(logits.detach() / 0.7, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1),
        ref_lp,
        adv,
        mask,
    )
    assert torch.all(seq_kl >= -1e-6)
    assert stats["ntok"] == float(mask.sum())

    # Compare lambda = 0 vs lambda = 0.1 objective and gradient norm decomposition
    loss0, info0 = compute_grpo_objective(
        logits,
        target_ids,
        ref_lp,
        adv,
        mask,
        lambda_entropy=0.0,
        with_entropy_reg=True,
        params_for_ratio=list(head.parameters()),
    )
    loss1, info1 = compute_grpo_objective(
        logits,
        target_ids,
        ref_lp,
        adv,
        mask,
        lambda_entropy=0.1,
        with_entropy_reg=True,
        params_for_ratio=list(head.parameters()),
    )
    assert info0["ent_grad_ratio"] == 0.0
    assert info1["gn_pg"] > 0 and info1["gn_H_reg"] > 0
    assert info1["ent_grad_ratio"] == pytest.approx(0.1 * info1["gn_H_reg"] / info1["gn_pg"])
    assert (loss0 - loss1).item() == pytest.approx(0.1 * info1["H_reg"], abs=1e-5)

    with pytest.raises(ValueError):
        compute_grpo_objective(logits, target_ids, ref_lp, adv, mask, lambda_entropy=0.05, with_entropy_reg=False)
