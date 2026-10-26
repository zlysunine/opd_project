"""Full-vocabulary T=1 diagnostics on a small, fixed response-position set."""
import numpy as np
import torch


def selected_log_probs(model, hidden, indices, chunk_size=16):
    if not indices:
        return np.empty((0, model.config.vocab_size), dtype=np.float32)
    head = model.get_output_embeddings()
    result = []
    for start in range(0, len(indices), chunk_size):
        selected = hidden[indices[start:start+chunk_size]]
        result.append(head(selected).float().log_softmax(-1).cpu().numpy())
    return np.concatenate(result)


def nll_bits(model, hidden, response, prediction_indices, chunk_size=16):
    if not response:
        return np.nan
    total = 0.
    for start in range(0, len(response), chunk_size):
        ix = prediction_indices[start:start+chunk_size]
        logp = model.get_output_embeddings()(hidden[ix]).float().log_softmax(-1)
        targets = torch.tensor(response[start:start+chunk_size], device=logp.device)
        total -= logp.gather(-1, targets[:, None]).sum().item()
    return total/len(response)/np.log(2.)


def distribution_metrics(logp, logq, initial=None, ks=(10,50,100)):
    p, q = np.exp(logp.astype(np.float64)), np.exp(logq.astype(np.float64))
    result = dict(student_entropy_nats=float(-(p*logp).sum()), teacher_entropy_nats=float(-(q*logq).sum()),
                  kl_student_teacher_nats=float((p*(logp-logq)).sum()))
    result['entropy_gap_signed_nats'] = result['teacher_entropy_nats']-result['student_entropy_nats']
    result['entropy_gap_abs_nats'] = abs(result['entropy_gap_signed_nats'])
    if initial is not None:
        result['kl_to_initial_nats'] = float((p*(logp-initial)).sum())
    # Stable full sort: tied probabilities use token ID. No support renormalization.
    a, b = np.argsort(-logp, kind='stable'), np.argsort(-logq, kind='stable')
    for k in ks:
        if not 1 <= k <= len(p):
            for suffix in ('overlap','shared_mass_student','shared_mass_teacher'):
                result[f'token_top{k}_{suffix}'] = np.nan
            continue
        shared = np.intersect1d(a[:k], b[:k])
        result[f'token_top{k}_overlap'] = len(shared)/k
        result[f'token_top{k}_shared_mass_student'] = float(p[shared].sum())
        result[f'token_top{k}_shared_mass_teacher'] = float(q[shared].sum())
    return result
