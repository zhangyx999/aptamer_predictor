"""Feature extraction: k-mer frequency vectors and RDKit molecular descriptors."""

from __future__ import annotations

from itertools import product

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


# ---------------------------------------------------------------------------
# Aptamer k-mer features
# ---------------------------------------------------------------------------

def rna_to_dna(sequence: str) -> str:
    """Convert RNA sequence to DNA form by replacing U with T."""
    return sequence.replace("U", "T").replace("u", "t")


def _get_all_kmers(k: int) -> list[str]:
    """Return all possible k-mers in lexicographic order (A, T, G, C)."""
    bases = ["A", "T", "G", "C"]
    return ["".join(p) for p in product(bases, repeat=k)]


def kmer_frequency(sequence: str, k: int) -> list[float]:
    """Compute normalised k-mer frequency vector for a single k.

    Returns a list of length 4^k in the same order as _get_all_kmers(k).
    """
    sequence = sequence.upper()
    all_kmers = _get_all_kmers(k)
    n_kmers = len(sequence) - k + 1
    if n_kmers <= 0:
        return [0.0] * len(all_kmers)

    # Count occurrences
    counts = {km: 0 for km in all_kmers}
    for i in range(n_kmers):
        sub = sequence[i: i + k]
        if sub in counts:
            counts[sub] += 1

    return [counts[km] / n_kmers for km in all_kmers]


def kmer_features(sequence: str, k_list: list[int]) -> list[float]:
    """Concatenated k-mer frequency vectors for multiple k values.

    Order: k_list[0] features, k_list[1] features, ...
    Example: k_list=[1,2,3,4] -> 4 + 16 + 64 + 256 = 340 features.
    """
    sequence = rna_to_dna(sequence)
    features = []
    for k in k_list:
        features.extend(kmer_frequency(sequence, k))
    return features


# ---------------------------------------------------------------------------
# Molecular descriptors
# ---------------------------------------------------------------------------

# Pre-compute descriptor list with Ipc excluded (210 - 1 = 209 descriptors)
_DESCRIPTOR_FUNCS = [
    (name, func) for name, func in Descriptors.descList if name != "Ipc"
]


def molecular_descriptors(smiles: str) -> list[float]:
    """Calculate 209 RDKit molecular descriptors (excluding Ipc).

    Returns a list of floats. Invalid SMILES returns a NaN-filled list.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [float("nan")] * len(_DESCRIPTOR_FUNCS)

    values = []
    for _name, func in _DESCRIPTOR_FUNCS:
        try:
            v = func(mol)
            if v is None:
                v = float("nan")
            values.append(float(v))
        except Exception:
            values.append(float("nan"))
    return values


def descriptor_names() -> list[str]:
    """Return the 209 descriptor names (excluding Ipc) in order."""
    return [name for name, _ in _DESCRIPTOR_FUNCS]


# ---------------------------------------------------------------------------
# Combined feature vector
# ---------------------------------------------------------------------------

# Mapping from model mer-label to the k values it uses
MER_K_MAP = {
    "1mer":  [1],
    "2mer":  [2],
    "4mer":  [4],
    "23mer": [2, 3],
    "24mer": [2, 4],
    "123mer": [1, 2, 3],
    "124mer": [1, 2, 4],
    "1234mer": [1, 2, 3, 4],
}


def build_feature_vector(sequence: str, smiles: str, k_list: list[int]) -> np.ndarray:
    """Build a complete feature vector: concatenated k-mer + 209 descriptors.

    Args:
        sequence: Aptamer sequence (RNA or DNA).
        smiles: Small molecule SMILES string.
        k_list: List of k values for k-mer features, e.g. [1, 2, 4].

    Returns:
        1-D numpy array of shape (sum(4^k for k in k_list) + 209,).
    """
    kmer = kmer_features(sequence, k_list)
    desc = molecular_descriptors(smiles)
    return np.array(kmer + desc, dtype=np.float64)
