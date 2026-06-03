import numpy as np

LUNAR_LOW = np.array([-2.5, -2.5, -10.0, -10.0, -6.2831855, -10.0, 0.0, 0.0])
LUNAR_HIGH = np.array([2.5, 2.5, 10.0, 10.0, 6.2831855, 10.0, 1.0, 1.0])


def make_bin_edges(low=LUNAR_LOW, high=LUNAR_HIGH, n_bins=7):
    return [np.linspace(lo, hi, n_bins + 1)[1:-1] for lo, hi in zip(low, high)]


def lunar_featurizer(edges=None, binary_dims=(6, 7), n_bins=7):
    if edges is None:
        edges = make_bin_edges(n_bins=n_bins)
    def featurize(obs):
        key = []
        for i, x in enumerate(obs):
            if i in binary_dims:
                key.append(int(round(float(x))))
            else:
                key.append(int(np.digitize(x, edges[i])))
        return tuple(key)
    return featurize


def constant_featurizer(state=0):
    def featurize(obs):
        return state
    return featurize


if __name__ == "__main__":
    f = lunar_featurizer()
    print(f(LUNAR_LOW), f(LUNAR_HIGH))
