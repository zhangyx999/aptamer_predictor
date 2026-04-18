"""Feature extraction: k-mer frequency vectors and RDKit molecular descriptors."""

from __future__ import annotations

from itertools import product

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from aptamer_predictor.descriptor_schema import TRAINING_DESCRIPTOR_NAMES


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

_DESCRIPTOR_FUNC_BY_NAME = dict(Descriptors.descList)
_MISSING_DESCRIPTOR_NAMES = [
    name for name in TRAINING_DESCRIPTOR_NAMES if name not in _DESCRIPTOR_FUNC_BY_NAME
]
if _MISSING_DESCRIPTOR_NAMES:
    missing = ", ".join(_MISSING_DESCRIPTOR_NAMES)
    raise RuntimeError(
        "Installed RDKit is missing descriptors required by the trained models: "
        f"{missing}"
    )

# The model schema intentionally excludes Ipc and any descriptors added in
# newer RDKit releases. Only these canonical training-time names are allowed.
_DESCRIPTOR_FUNCS = [
    (name, _DESCRIPTOR_FUNC_BY_NAME[name]) for name in TRAINING_DESCRIPTOR_NAMES
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


def build_feature_vector_fast(
    sequence: str,
    precomputed_desc: list[float],
    k_list: list[int],
) -> np.ndarray:
    """Build feature vector using pre-computed molecular descriptors.

    Avoids redundant RDKit computation when all mutants share one SMILES.
    """
    kmer = kmer_features(sequence, k_list)
    vec = np.array(kmer + precomputed_desc, dtype=np.float64)
    return np.nan_to_num(vec, nan=0.0)


# ---------------------------------------------------------------------------
# Vectorized batch feature matrix (for mutation enumeration)
# ---------------------------------------------------------------------------

_BASE_MAP = {"A": 0, "T": 1, "G": 2, "C": 3}
_ENCODE_TABLE = np.zeros(256, dtype=np.int32)
_ENCODE_TABLE[ord("A")] = 0
_ENCODE_TABLE[ord("a")] = 0
_ENCODE_TABLE[ord("T")] = 1
_ENCODE_TABLE[ord("t")] = 1
_ENCODE_TABLE[ord("U")] = 1
_ENCODE_TABLE[ord("u")] = 1
_ENCODE_TABLE[ord("G")] = 2
_ENCODE_TABLE[ord("g")] = 2
_ENCODE_TABLE[ord("C")] = 3
_ENCODE_TABLE[ord("c")] = 3


def _encode_sequence(sequence: str) -> np.ndarray:
    """Encode DNA sequence as int array.  A=0 T=1 G=2 C=3, unknown→0."""
    return np.array([_BASE_MAP.get(c, 0) for c in sequence], dtype=np.int32)


def _encode_sequences(sequences: list[str]) -> np.ndarray:
    if not sequences:
        return np.empty((0, 0), dtype=np.int32)

    length = len(sequences[0])
    joined = "".join(sequences).encode("ascii")
    encoded = _ENCODE_TABLE[np.frombuffer(joined, dtype=np.uint8)]
    return encoded.reshape(len(sequences), length)


def build_feature_matrix(
    sequences: list[str] | np.ndarray,
    precomputed_desc: list[float],
    k_list: list[int],
) -> np.ndarray:
    """Vectorized batch feature matrix construction.

    All sequences **must** have the same length.  Molecular descriptors are
    tiled across the batch so RDKit is called only once.

    Returns ndarray of shape ``(n_sequences, total_feature_dim)`` with NaN→0.
    """
    if isinstance(sequences, np.ndarray):
        if sequences.size == 0:
            return np.empty((0, 0))
        if sequences.ndim != 2:
            raise ValueError("Encoded sequence array must be 2-D")

        if np.issubdtype(sequences.dtype, np.integer) and (
            sequences.size == 0
            or (int(np.min(sequences)) >= 0 and int(np.max(sequences)) <= 3)
        ):
            encoded = sequences.astype(np.int32, copy=False)
        else:
            encoded = _ENCODE_TABLE[sequences.astype(np.uint8, copy=False)]
    else:
        if not sequences:
            return np.empty((0, 0))
        encoded = _encode_sequences(sequences)

    N = encoded.shape[0]
    desc_arr = np.array(precomputed_desc, dtype=np.float64)

    L = encoded.shape[1]

    all_kmer = []
    for k in k_list:
        dim = 4 ** k
        n_kmers = L - k + 1
        if n_kmers <= 0:
            all_kmer.append(np.zeros((N, dim), dtype=np.float64))
            continue

        # k-mer index = base-4 number at each sliding-window position
        indices = np.zeros((N, n_kmers), dtype=np.int32)
        for i in range(k):
            indices = indices * 4 + encoded[:, i: i + n_kmers]

        # Single-call batch bincount via offset trick
        offsets = np.arange(N, dtype=np.int32)[:, None] * dim
        flat = (indices + offsets).ravel()
        counts = np.bincount(flat, minlength=N * dim).reshape(N, dim).astype(np.float64)
        counts /= n_kmers
        all_kmer.append(counts)

    kmer_matrix = np.hstack(all_kmer)  # (N, total_kmer_dim)
    desc_matrix = np.tile(desc_arr, (N, 1))  # (N, 209)
    result = np.hstack([kmer_matrix, desc_matrix])
    return np.nan_to_num(result, nan=0.0)


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
    vec = np.array(kmer + desc, dtype=np.float64)
    # Replace NaN with 0 (matches original pipeline: "non" → "0")
    # This is critical for PyTorch RNN/biRNN models which propagate NaN
    vec = np.nan_to_num(vec, nan=0.0)
    return vec
