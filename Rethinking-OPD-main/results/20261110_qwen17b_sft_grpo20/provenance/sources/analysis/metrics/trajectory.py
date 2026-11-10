"""Response-local curvature in degrees; no fabricated angles across special tokens."""
import numpy as np


def curvature(h, token_mask=None, eps=1e-8):
    # Cast BEFORE differencing (BF16/FP16 differences otherwise lose small motion).
    h = np.asarray(h, dtype=np.float32)
    mask = np.ones(len(h), dtype=bool) if token_mask is None else np.asarray(token_mask, dtype=bool)
    if mask.shape != (len(h),):
        raise ValueError('One mask entry per response token is required')
    if len(h) < 3:
        return dict(mean_deg=np.nan, angles_deg=np.empty(0), valid_angles=0,
                    invalid_fraction=np.nan, mean_step_norm=np.nan)
    v = np.diff(h, axis=0)
    norms = np.linalg.norm(v, axis=1)
    triples = mask[:-2] & mask[1:-1] & mask[2:]
    good = triples & np.isfinite(norms[:-1]) & np.isfinite(norms[1:])
    good &= (norms[:-1] > eps) & (norms[1:] > eps)
    angles = np.full(len(h)-2, np.nan)
    dots = np.einsum('ij,ij->i', v[:-1][good], v[1:][good])
    angles[good] = np.degrees(np.arccos(np.clip(dots/(norms[:-1][good]*norms[1:][good]), -1, 1)))
    steps = mask[:-1] & mask[1:] & np.isfinite(norms)
    return dict(mean_deg=float(angles[good].mean()) if good.any() else np.nan, angles_deg=angles,
                valid_angles=int(good.sum()),
                invalid_fraction=float(1-good.sum()/triples.sum()) if triples.any() else np.nan,
                mean_step_norm=float(norms[steps].mean()) if steps.any() else np.nan)


def pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    value = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan
    return value, int(valid.sum())


def paired_difference(current, target):
    current, target = np.asarray(current, dtype=float), np.asarray(target, dtype=float)
    valid = np.isfinite(current) & np.isfinite(target)
    return (float(np.mean(current[valid]-target[valid])) if valid.any() else np.nan, int(valid.sum()))
