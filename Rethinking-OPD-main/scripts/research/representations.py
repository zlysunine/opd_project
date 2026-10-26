"""Within-scale aligned activation diagnostics; cross-dimensional CKA/RSA are explicit."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 1e-12), norms[..., 0] > 1e-12


def cosine(a, b):
    if a.shape != b.shape:
        raise ValueError('Coordinate cosine requires identical aligned dimensions and rows')
    an, av = normalize(a)
    bn, bv = normalize(b)
    return np.where(av & bv, (an*bn).sum(-1), np.nan)


def rsa(x, max_samples=2048):
    if len(x) > max_samples:
        raise ValueError(f'RSA O(N^2) cap exceeded: {len(x)}>{max_samples}')
    xn, valid = normalize(x)
    r = xn @ xn.T
    r[~valid, :] = np.nan
    r[:, ~valid] = np.nan
    return r


def rsa_correlations(a, b):
    if a.shape != b.shape:
        raise ValueError('RSA matrices must describe the same samples')
    ix = np.triu_indices(len(a), 1)
    av, bv = a[ix], b[ix]
    mask = np.isfinite(av) & np.isfinite(bv)
    av, bv = av[mask], bv[mask]
    if len(av) < 3 or np.std(av) < 1e-12 or np.std(bv) < 1e-12:
        return {'pearson_upper_triangle': None, 'spearman_upper_triangle': None}
    return {'pearson_upper_triangle': float(np.corrcoef(av, bv)[0, 1]),
            'spearman_upper_triangle': float(spearmanr(av, bv).statistic)}


def neighborhood_overlap(a, b, k, distance='cosine', max_samples=2048):
    if len(a) != len(b) or not 1 <= k < len(a) or len(a) > max_samples:
        raise ValueError('Neighborhoods need matched rows, 1<=k<N and N<=cap')
    def neighbors(x):
        if distance == 'cosine':
            xn, valid = normalize(x)
            if not valid.all():
                raise ValueError('Cosine neighborhood undefined for zero vectors')
            d = 1-xn@xn.T
        elif distance == 'euclidean':
            d = np.maximum(0, (x*x).sum(-1)[:, None] + (x*x).sum(-1)[None, :] - 2*x@x.T)
        else:
            raise ValueError(distance)
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, axis=1, kind='stable')[:, :k]
    na, nb = neighbors(a), neighbors(b)
    return {'representation_neighborhood_overlap': float(np.mean([len(set(x)&set(y))/k for x, y in zip(na, nb)])),
            'n': len(a), 'k': k, 'random_reference': k/(len(a)-1), 'distance': distance}


def linear_cka(x, y):
    if len(x) != len(y):
        raise ValueError('CKA requires aligned sample rows')
    x, y = x-x.mean(0), y-y.mean(0)
    # Sample Gram form is exactly the centered linear CKA formula; supports d_A != d_B.
    k, l = x@x.T, y@y.T
    denom = np.linalg.norm(k)*np.linalg.norm(l)
    return float(np.sum(k*l)/denom) if denom > 1e-12 else None


def subspace_overlap(a, b, rank, centered=False):
    if a.shape[1] != b.shape[1]:
        raise ValueError('Subspace overlap requires the same feature coordinate system')
    if centered:
        a, b = a-a.mean(0), b-b.mean(0)
    if rank < 1 or rank > min(np.linalg.matrix_rank(a), np.linalg.matrix_rank(b)):
        raise ValueError('Requested equal rank exceeds a change matrix rank')
    # Rows are sample changes, so RIGHT singular vectors span feature-space changes.
    va = np.linalg.svd(a, full_matrices=False)[2][:rank].T
    vb = np.linalg.svd(b, full_matrices=False)[2][:rank].T
    return {'representation_subspace_overlap': float(np.linalg.norm(va.T@vb)**2/rank),
            'rank': rank, 'decomposition': 'centered_PCA' if centered else 'uncentered_SVD'}


def clustering_ari(x, labels, clusters, seed):
    if not 1 <= clusters <= len(x):
        raise ValueError('Invalid K-means cluster count')
    predicted = KMeans(n_clusters=clusters, random_state=seed, n_init=10).fit_predict(x)
    return {'method': 'seeded_KMeans_engineering_baseline_not_ADP', 'clusters': clusters, 'seed': seed,
            'ari': {key: float(adjusted_rand_score(values, predicted)) for key, values in labels.items()}}


def effective_rank(x, spectrum='singular_values'):
    s = np.linalg.svd(x, compute_uv=False)
    if spectrum == 'squared_singular_values':
        s = s*s
    elif spectrum != 'singular_values':
        raise ValueError(spectrum)
    if s.sum() == 0:
        return None
    p = s[s>0]/s.sum()
    return float(np.exp(-np.sum(p*np.log(p))))


def state_probe(x, records, label='intermediate_state', seed=0):
    """Use dataset instance/template splits; never random neighboring token splits."""
    split = np.array([r['probe_split'] for r in records])
    y = np.array([str(r['labels'][label]) for r in records])
    for key in ['problem_id', 'instance_id', 'template_id']:
        sets = {s: {r[key] for r in records if r['probe_split']==s} for s in ('train', 'validation', 'test')}
        if sets['train'] & sets['test'] or sets['train'] & sets['validation'] or sets['validation'] & sets['test']:
            raise ValueError(f'Probe {key} leaks across splits')
    train, val, test = [split == s for s in ('train', 'validation', 'test')]
    if not all(mask.any() for mask in (train, val, test)) or len(set(y[train])) < 2:
        raise ValueError('Probe needs all splits and at least two training classes')
    counts = {v: int((y[train]==v).sum()) for v in set(y[train])}
    majority = max(counts, key=counts.get)
    best = None
    for c in (0.01, 0.1, 1., 10.):
        model = LogisticRegression(C=c, max_iter=1000, random_state=seed).fit(x[train], y[train])
        acc = model.score(x[val], y[val])
        if best is None or acc > best[0]:
            best = (acc, c, model)
    return {'label': label, 'validation_accuracy': best[0], 'C_selected_on_validation': best[1],
            'held_out_accuracy': best[2].score(x[test], y[test]), 'majority_baseline': float(np.mean(y[test]==majority)),
            'train_classes': counts, 'test_classes': {v: int((y[test]==v).sum()) for v in set(y[test])},
            'n_train': int(train.sum()), 'n_validation': int(val.sum()), 'n_test': int(test.sum()),
            'interpretation': 'decodability is not evidence of causal use'}
