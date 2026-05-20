import numpy as np

class SpatialMask:
    """Apply 3D spatial distance mask to CCMPred coupling parameters.

    Kernel formulas:
        sigmoid:  W(d) = 1 / (1 + exp(gamma * (d - d0)))
        hill:     W(d) = 1 / (1 + (d / d0)^n)

    d_ij = Cβ-Cβ distance (Cα for GLY) in Angstrom
    d0   = 8.0 Å (CASP gold standard, LOCK this for Cβ)
    """

    def __init__(self, distance_matrix, d0=8.0, gamma=1.5, mode='multiply', epsilon=0.05,
                 alpha=1.0, kernel='sigmoid', n=4):
        self.D = np.asarray(distance_matrix, dtype=np.float32)
        self.d0 = d0
        self.gamma = gamma
        self.mode = mode
        self.epsilon = epsilon
        self.alpha = alpha
        self.kernel = kernel
        self.n = n
        self._mask = None

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self._compute()
        return self._mask

    def _compute(self):
        if self.kernel == 'hill':
            W = 1.0 / (1.0 + (self.D / self.d0) ** self.n)
        else:
            W = 1.0 / (1.0 + np.exp(self.gamma * (self.D - self.d0)))
        np.fill_diagonal(W, 0.0)
        return W.astype(np.float32)

    def apply_to_eij(self, eij):
        """Apply spatial mask to CCMPred eij coupling table.

        eij: (L, L, 21, 21) array
        mode='multiply':  eij_new = W * eij  (attenuate distant pairs)
        mode='divide':    eij_new = eij / (W + epsilon)  (boost close pairs)
        mode='surprise':  divide + long-range reward
            eij_norm(i,j) = ||eij[i,j]||_F / max_norm
            reward = 1 + alpha * (1 - W) * tanh(eij_norm)
            eij_new = eij / (W + epsilon) * reward
        """
        W = self.mask[:, :, None, None]
        if self.mode == 'multiply':
            return eij * W
        elif self.mode == 'divide':
            return eij / (W + self.epsilon)
        elif self.mode == 'surprise':
            # Frobenius norm per residue pair → (L, L)
            eij_norm = np.sqrt(np.sum(eij ** 2, axis=(2, 3)))
            max_norm = eij_norm.max()
            if max_norm > 0:
                eij_norm = eij_norm / max_norm
            W_2d = self.mask  # (L, L)
            reward = 1.0 + self.alpha * (1.0 - W_2d) * np.tanh(eij_norm)
            reward = reward[:, :, None, None]
            return eij / (W + self.epsilon) * reward
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

    def stats(self):
        """Return mask statistics for logging."""
        W = self.mask
        n_pairs = W.size - W.shape[0]
        return {
            'd0': self.d0,
            'gamma': self.gamma,
            'mean_weight': float(W[W > 0].mean()),
            'pairs_retained_05': int((W > 0.5).sum()) - W.shape[0],
            'pairs_retained_01': int((W > 0.1).sum()) - W.shape[0],
            'total_pairs': int(n_pairs),
        }
