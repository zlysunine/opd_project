"""Raw Euclidean kNN, excluding self; distance ties use sample ID order."""
import numpy as np


def knn(h, sample_ids, k):
    h = np.asarray(h, dtype=np.float64)
    if len(set(sample_ids)) != len(h) or not 0 < k < len(h) or not np.isfinite(h).all():
        raise ValueError('kNN needs unique IDs, finite raw vectors and 0<k<N')
    # Stable sort after ID-ordering gives deterministic tie breaks independent of row order.
    order = np.argsort(np.asarray(sample_ids), kind='stable')
    norms = (h*h).sum(1)
    dist = np.sqrt(np.maximum(0, norms[:, None]+norms[None, :]-2*h@h.T))
    np.fill_diagonal(dist, np.inf)
    indices = order[np.argsort(dist[:, order], axis=1, kind='stable')[:, :k]]
    return indices, np.take_along_axis(dist, indices, axis=1)


def neighbor_overlap(current, initial):
    if current.shape != initial.shape or current.shape[1] == 0:
        raise ValueError('Matched nonempty neighborhood arrays required')
    return np.asarray([len(set(a) & set(b))/len(a) for a, b in zip(current, initial)])
