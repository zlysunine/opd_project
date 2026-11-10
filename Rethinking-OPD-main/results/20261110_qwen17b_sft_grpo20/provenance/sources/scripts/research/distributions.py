"""Full-vocabulary diagnostics, computed in response-position chunks only."""
import math
import numpy as np
import torch


@torch.no_grad()
def distribution_metrics(student_logits, teacher_logits, ks=(16,), temperature=1.0):
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError('Expected matched [positions, vocabulary] logits')
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError('diagnostic_temperature must be finite and positive')
    p = (student_logits.double() / temperature).log_softmax(-1)
    q = (teacher_logits.double() / temperature).log_softmax(-1)
    if not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError('Nonfinite logits/log probabilities')
    hp, hq = -(p.exp()*p).sum(-1), -(q.exp()*q).sum(-1)
    result = {'entropy/student_full_vocab': hp, 'entropy/teacher_full_vocab': hq,
              'entropy/gap_absolute': (hq-hp).abs(), 'entropy/gap_signed': hq-hp,
              'kl/full_vocab_p_to_q': (p.exp()*(p-q)).sum(-1),
              'kl/full_vocab_q_to_p': (q.exp()*(q-p)).sum(-1)}
    for k in ks:
        if not 1 <= k <= p.shape[-1]:
            raise ValueError(f'Invalid diagnostic k={k} for vocabulary={p.shape[-1]}')
        sp, sq = p.topk(k, -1).indices, q.topk(k, -1).indices
        shared = (sp.unsqueeze(-1) == sq.unsqueeze(-2)).any(-1)
        n = shared.sum(-1)
        pl, ql = p.gather(-1, sp), q.gather(-1, sp)
        pi, qi = pl.masked_fill(~shared, -torch.inf), ql.masked_fill(~shared, -torch.inf)
        # Work only on nonempty intersections: empty rows remain NaN + valid mask.
        adv = torch.full_like(hp, torch.nan)
        valid = n > 0
        pp = pi[valid] - pi[valid].logsumexp(-1, keepdim=True)
        qq = qi[valid] - qi[valid].logsumexp(-1, keepdim=True)
        ratio = torch.where(shared[valid], qq-pp, 0.)
        adv[valid] = (pp.exp()*ratio).sum(-1) / n[valid]
        pb, qb = pl.log_softmax(-1), ql.log_softmax(-1)
        result.update({f'token_topk/{k}/overlap_ratio': n.double()/k,
                       f'token_topk/{k}/overlap_advantage': adv,
                       f'token_topk/{k}/advantage_valid': valid.double(),
                       f'token_topk/{k}/empty_intersection_rate': (~valid).double(),
                       f'token_topk/{k}/shared_mass_student': (pl.exp()*shared).sum(-1),
                       f'token_topk/{k}/shared_mass_teacher': (ql.exp()*shared).sum(-1),
                       f'kl/renormalized_student_topk_{k}_p_to_q': (pb.exp()*(pb-qb)).sum(-1),
                       f'loss/repo_raw_log_ratio_student_topk_{k}': (pl.softmax(-1)*(pl-ql)).sum(-1)})
    return {key: value.cpu().numpy() for key, value in result.items()}


def aggregate(rows, value_key='value'):
    """Token and equal-question means; trajectories within question share its weight."""
    by_metric = {}
    for row in rows:
        by_metric.setdefault(row['metric'], []).append(row)
    out = {}
    for metric, group in by_metric.items():
        valid = [r for r in group if r[value_key] is not None and math.isfinite(r[value_key])]
        questions = {}
        for row in valid:
            questions.setdefault(row['problem_id'], []).append(row[value_key])
        out[metric] = {'token_weighted': float(np.mean([r[value_key] for r in valid])) if valid else None,
                       'question_mean': float(np.mean([np.mean(v) for v in questions.values()])) if questions else None,
                       'valid_tokens': len(valid), 'total_tokens': len(group), 'questions': len(questions)}
    return out


def pass_at_k(n, c, k):
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError('pass@k requires 0<=c<=n and n>=k>=1')
    if n-c < k:
        return 1.0
    return -math.expm1(sum(math.log1p(-c/(n-i)) for i in range(k)))


def evaluate_samples(rows, ks=(1, 16)):
    groups = {}
    benchmarks = {}
    for row in rows:
        groups.setdefault(row['problem_id'], []).append(row)
    if not groups:
        raise ValueError('No evaluation samples')
    output = {'question_count': len(groups), 'aggregation': 'equal questions within benchmark; macro benchmarks',
              'per_question': {}, 'privileged': any(r.get('privileged', False) for r in rows)}
    for pid, group in groups.items():
        n, c = len(group), sum(int(r['correct']) for r in group)
        scores = {'n': n, 'correct': c, 'avg_accuracy': c/n,
                  **{f'pass@{k}': pass_at_k(n, c, k) for k in ks if k <= n}}
        if n == 16:
            scores['avg@16'] = c/16
        output['per_question'][pid] = scores
        benchmarks.setdefault(group[0].get('benchmark', 'default'), []).append(scores)
    names = set.intersection(*(set(x) for x in output['per_question'].values())) - {'n', 'correct'}
    output['benchmarks'] = {b: {key: float(np.mean([s[key] for s in v])) for key in names}
                            for b, v in benchmarks.items()}
    for key in names:
        output[key] = float(np.mean([b[key] for b in output['benchmarks'].values()]))
    output['truncation_rate'] = float(np.mean([r['truncated'] for r in rows]))
    return output


def gap_recovery(before, after, teacher, threshold=1e-6, teacher_checkpoint=None, privileged=False):
    if teacher_checkpoint is None:
        raise ValueError('Gap recovery requires a fixed reference teacher checkpoint')
    denom = teacher-before
    return {'gap_recovery': (after-before)/denom if denom > threshold else None,
            'applicable': denom > threshold, 'teacher_checkpoint': teacher_checkpoint,
            'teacher_uses_privilege': privileged, 'denominator': denom}
