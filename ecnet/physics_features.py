import numpy as np

# Kyte-Doolittle hydrophobicity scale (J Mol Biol 157:105-132, 1982)
KYTE_DOOLITTLE = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}

# Charge states at physiological pH 7.4
AA_CHARGE = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.5,
             'C': 0, 'Y': 0, 'S': 0, 'T': 0, 'N': 0, 'Q': 0,
             'A': 0, 'G': 0, 'V': 0, 'L': 0, 'I': 0, 'M': 0,
             'F': 0, 'W': 0, 'P': 0}

LAMBDA_DEBYE = 6.0   # Debye screening length (A) at ~150mM ionic strength


class PhysicsFeatures:
    """Compute physics proxy features from wild-type 3D structure.

    Three empirical features grounded in decades of protein thermodynamics:
      1. Contact Density  C_i  — proxy for solvent accessibility
      2. Hydrophobic Packing H_i — proxy for core burial energy
      3. Electrostatic Energy  E_coulomb — Debye-Huckel screened potential

    References:
      - Shakhnovich (1994) Phys Rev Lett
      - Kauzmann (1959) Adv Protein Chem
      - Eisenberg & McLachlan (1986) Nature
      - Tanford-Kirkwood (1957) JACS
      - Kumar & Nussinov (2002) Biophys J
    """

    def __init__(self, coords, sequence, contact_cutoff=8.0):
        """
        coords: (L, 3) numpy array of Cβ coordinates (Cα for GLY)
        sequence: length-L string of amino acid single-letter codes
        """
        self.coords = np.asarray(coords, dtype=np.float32)
        self.sequence = sequence
        self.L = len(sequence)
        self.cutoff = contact_cutoff

        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.D = np.sqrt(np.sum(diff ** 2, axis=-1))

        self._C = None
        self._H = None
        self._E = None

    @property
    def contact_density(self):
        """C_i = number of neighbours within 8A of residue i."""
        if self._C is None:
            contact = (self.D < self.cutoff).astype(np.float32)
            np.fill_diagonal(contact, 0)
            self._C = contact.sum(axis=1)  # (L,)
        return self._C

    @property
    def hydrophobic_packing(self):
        """H_i = Σ_j hydrophobicity_j * 1[d_ij < cutoff]."""
        if self._H is None:
            h = np.array([KYTE_DOOLITTLE[aa] for aa in self.sequence],
                         dtype=np.float32)
            contact = (self.D < self.cutoff).astype(np.float32)
            np.fill_diagonal(contact, 0)
            self._H = (contact * h[None, :]).sum(axis=1)  # (L,)
        return self._H

    @property
    def electrostatic_energy(self):
        """Debye-Huckel screened Coulomb energy between charged residues."""
        if self._E is None:
            q = np.array([AA_CHARGE.get(aa, 0) for aa in self.sequence],
                         dtype=np.float32)
            qij = q[:, None] * q[None, :]  # (L, L)
            with np.errstate(divide='ignore', invalid='ignore'):
                E = qij * np.exp(-self.D / LAMBDA_DEBYE) / (self.D + 1e-8)
                np.fill_diagonal(E, 0)
                E[~np.isfinite(E)] = 0
            self._E = E.sum()  # scalar
        return self._E

    @property
    def feature_vector(self):
        """Aggregated physics feature vector for predictor concatenation.

        Returns (6,) array:
          [mean_contact, max_contact, mean_hydrophobic, max_hydrophobic,
           electrostatic_total, contact_ratio]
        """
        C = self.contact_density
        H = self.hydrophobic_packing
        E = self.electrostatic_energy
        contact_ratio = (C > 0).sum() / self.L

        return np.array([
            float(C.mean()),
            float(C.max()),
            float(H.mean()),
            float(H.max()),
            float(E),
            float(contact_ratio),
        ], dtype=np.float32)
