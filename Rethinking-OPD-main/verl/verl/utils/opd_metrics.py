# Copyright 2026 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paper Section 2.3 diagnostics on already-computed student rollout tensors."""

import torch


@torch.no_grad()
def compute_opd_dynamic_metrics(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    overlap_mask: torch.Tensor,
    response_mask: torch.Tensor,
    student_entropy: torch.Tensor,
    teacher_entropy: torch.Tensor,
) -> dict[str, float]:
    """Return masked, position-weighted means, using natural logarithms.

    Log-probs and overlap_mask have shape [batch, response_length, k]. The
    teacher log-probs refer to the SAME student top-k token IDs, not the
    teacher's independently ordered top-k. Entropies have shape [batch, time]
    and must be full-vocabulary entropies at the same student-visited states.

    Overlap advantage is mean_t(-KL(p_I || q_I) / |I_t|), with both models
    normalized within I_t. It is undefined for empty intersections: those
    positions are excluded from that metric only, and their frequency is logged.
    Padding is excluded from every metric. No training reward is modified.
    """
    if student_log_probs.ndim != 3 or student_log_probs.shape[-1] == 0:
        raise ValueError("Expected nonempty [batch, response_length, k] log-probs")
    if teacher_log_probs.shape != student_log_probs.shape or overlap_mask.shape != student_log_probs.shape:
        raise ValueError("Student log-probs, teacher-on-student log-probs and overlap mask must have matching shapes")
    shape = student_log_probs.shape[:-1]
    if any(tensor.shape != shape for tensor in (response_mask, student_entropy, teacher_entropy)):
        raise ValueError("Response mask and full-vocabulary entropies must match [batch, response_length]")
    valid = response_mask.bool()
    count = int(valid.sum().item())
    metrics = {"opd/valid_response_tokens": float(count)}
    if count == 0:
        return metrics

    hs = student_entropy[valid].float()
    ht = teacher_entropy[valid].float()
    if not torch.isfinite(hs).all() or not torch.isfinite(ht).all():
        raise ValueError("Nonfinite entropy at a valid response position")

    intersection = overlap_mask.bool()[valid]
    cardinalities = intersection.sum(dim=-1)
    nonempty = cardinalities > 0
    metrics.update(
        {
            "opd/overlap_ratio": (cardinalities.float() / student_log_probs.shape[-1]).mean().item(),
            "opd/student_entropy": hs.mean().item(),
            "opd/teacher_entropy": ht.mean().item(),
            # E[|Hq - Hp|], rather than |E[Hq] - E[Hp]|.
            "opd/entropy_gap": (ht - hs).abs().mean().item(),
            "opd/entropy_gap_signed": (ht - hs).mean().item(),
            "opd/overlap_nonempty_fraction": nonempty.float().mean().item(),
            "opd/empty_intersection_rate": (~nonempty).float().mean().item(),
            "opd/shared_mass_student": (
                student_log_probs[valid].float().exp() * intersection
            ).sum(-1).mean().item(),
            "opd/shared_mass_teacher": (
                teacher_log_probs[valid].float().exp() * intersection
            ).sum(-1).mean().item(),
        }
    )
    # These use the queried student support, not the intersection-only advantage.
    s_all = student_log_probs[valid].float()
    t_all = teacher_log_probs[valid].float()
    if torch.isfinite(s_all).all() and torch.isfinite(t_all).all():
        p_subset = s_all.log_softmax(-1)
        q_subset = t_all.log_softmax(-1)
        metrics["opd/repo_raw_log_ratio_student_topk"] = (
            p_subset.exp() * (s_all - t_all)
        ).sum(-1).mean().item()
        metrics["opd/kl_renormalized_student_topk_p_to_q"] = (
            p_subset.exp() * (p_subset - q_subset)
        ).sum(-1).mean().item()
    if nonempty.any():
        shared = intersection[nonempty]
        s = student_log_probs[valid][nonempty].float()
        t = teacher_log_probs[valid][nonempty].float()
        if not torch.isfinite(s[shared]).all() or not torch.isfinite(t[shared]).all():
            raise ValueError("Nonfinite log-probability for a shared top-k token")
        s = s.masked_fill(~shared, -torch.inf)
        t = t.masked_fill(~shared, -torch.inf)
        log_p = s - torch.logsumexp(s, dim=-1, keepdim=True)
        log_q = t - torch.logsumexp(t, dim=-1, keepdim=True)
        # Off-intersection entries are -inf in both distributions; mask the
        # difference before multiplication to avoid 0 * nan.
        log_ratio = torch.where(shared, log_q - log_p, 0.0)
        per_position = (log_p.exp() * log_ratio).sum(dim=-1) / cardinalities[nonempty]
        metrics["opd/overlap_token_advantage"] = per_position.mean().item()
    return metrics
