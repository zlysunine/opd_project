"""Participation ratio is a variance statistic, not Gride intrinsic dimension."""
import numpy as np


def spectrum(h):
    h = np.asarray(h, dtype=np.float64)
    if not len(h) or not np.isfinite(h).all():
        return dict(participation_ratio=np.nan, top_eigenvalue_fraction=np.nan, activation_norm=np.nan)
    x = h-h.mean(0, keepdims=True)
    # N usually much smaller than hidden_size; avoid a d x d covariance.
    lam = np.maximum(0, np.linalg.eigvalsh(x@x.T))
    den = np.sum(lam**2)
    return dict(participation_ratio=float(lam.sum()**2/den) if den > 0 else np.nan,
                top_eigenvalue_fraction=float(lam[-1]/lam.sum()) if lam.sum() > 0 else np.nan,
                activation_norm=float(np.linalg.norm(h, axis=1).mean()))


def linear_cka(h, h0):
    x, y = h-h.mean(0), h0-h0.mean(0)
    a, b = x@x.T, y@y.T
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float((a*b).sum()/den) if den > 0 else np.nan
