import numpy as np

def kfold_split_edges(edges, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(edges))
    rng.shuffle(idx)
    folds = np.array_split(idx, k)

    out = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((edges[train_idx], edges[test_idx]))
    return out


def remove_edges_from_adj(A, edges):
    A2 = A.copy()
    A2[edges[:, 0], edges[:, 1]] = 0
    return A2


def negative_sample(A_forbidden, num_neg, seed=42, max_tries=5000):
    rng = np.random.default_rng(seed)
    n_dis, n_drug = A_forbidden.shape

    neg_edges = []
    seen = set()

    tries = 0
    while len(neg_edges) < num_neg:
        if tries > max_tries:
            raise RuntimeError(
                f"negative_sample: too many tries; collected={len(neg_edges)}/{num_neg}. "
                f"Graph too dense or forbidden matrix incorrect."
            )
        tries += 1

        remain = num_neg - len(neg_edges)
        batch = max(remain * 2, 1000)

        ds = rng.integers(0, n_dis, size=batch, endpoint=False)
        rs = rng.integers(0, n_drug, size=batch, endpoint=False)

        for d, r in zip(ds, rs):
            if A_forbidden[d, r] == 0:
                key = int(d) * n_drug + int(r)
                if key not in seen:
                    seen.add(key)
                    neg_edges.append((int(d), int(r)))
            if len(neg_edges) >= num_neg:
                break

    return np.array(neg_edges, dtype=int)