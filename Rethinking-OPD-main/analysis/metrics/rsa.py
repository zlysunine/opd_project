"""Cosine representational similarity, with explicit invalid vectors and paired IDs."""
import numpy as np
from scipy.stats import spearmanr


def align_rows(values, sample_ids, expected_ids):
    if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != set(expected_ids):
        raise ValueError('Representations must have the same unique sample IDs')
    lookup = {sid: i for i, sid in enumerate(sample_ids)}
    return np.asarray(values)[[lookup[sid] for sid in expected_ids]]


def cosine_rsm(h):
    h = np.asarray(h, dtype=np.float64)
    norms = np.linalg.norm(h, axis=1)
    valid = np.isfinite(h).all(1) & (norms > 0)
    u = np.zeros_like(h)
    u[valid] = h[valid] / norms[valid, None]
    r = np.clip(u @ u.T, -1, 1)
    r[~valid, :] = np.nan
    r[:, ~valid] = np.nan
    return r, valid


def rsm_summary(r, topics, bins=40):
    ii = np.triu_indices(len(r), 1)
    v = r[ii]
    good = np.isfinite(v)
    values = v[good]
    same = np.asarray(topics)[ii[0]] == np.asarray(topics)[ii[1]]
    avg = lambda x: float(np.mean(x)) if len(x) else np.nan
    within, between = avg(v[good & same]), avg(v[good & ~same])
    if len(set(topics)) < 2:
        within, between = np.nan, np.nan
    hist, edges = np.histogram(values, bins=np.linspace(-1, 1, bins + 1))
    stats = {'rsm_mean': avg(values), 'rsm_std': float(np.std(values)) if len(values) else np.nan,
             'rsm_topic_within': within, 'rsm_topic_between': between, 'rsm_topic_gap': within - between}
    for q in (5, 50, 95):
        stats[f'rsm_q{q}'] = float(np.percentile(values, q)) if len(values) else np.nan
    return stats, hist, edges


def compare_rsms(r, r0, max_pairs=None, seed=0):
    """Compare aligned RSMs without assuming a temporal or model-role baseline."""
    if r.shape != r0.shape or r.ndim!=2 or r.shape[0]!=r.shape[1]:
        raise ValueError('RSM shapes/sample sets must match')
    ii = np.triu_indices(len(r), 1)
    if max_pairs and len(ii[0]) > max_pairs:
        chosen = np.sort(np.random.default_rng(seed).choice(len(ii[0]), max_pairs, replace=False))
        ii = (ii[0][chosen], ii[1][chosen])
    a, b = r[ii], r0[ii]
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    rho = np.nan
    if len(a) >= 2 and np.ptp(a) > 0 and np.ptp(b) > 0:
        rho = float(spearmanr(a, b).statistic)
    return {'rho': rho,
            'rmse': float(np.sqrt(np.mean((a-b)**2))) if len(a) else np.nan,
            'n_pairs': len(a)}


def rsa_to_initial(r, r0, max_pairs=None, seed=0):
    result=compare_rsms(r,r0,max_pairs,seed)
    return {'rsa_rho_to_initial':result['rho'], 'rsa_rmse_to_initial':result['rmse'],
            'n_pairs':result['n_pairs']}
