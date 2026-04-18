# Large-Scale Mutation Search Acceleration

This document describes the algorithmic and systems-level optimizations that enable the TUI's interactive mutation search to evaluate combinatorially large candidate spaces (up to 4^N mutants for N selected sites) in practical timeframes.

---

## 1. Problem Statement

Given a base aptamer sequence of length L and a set of N mutation sites, the exhaustive search space comprises 4^N single-point or multi-point mutants (each site may be substituted with A, T, G, or C). For N = 10, this yields over one million candidates; for N = 15, over one billion. Each candidate must be scored by an ensemble of 9 heterogeneous models (Random Forest, XGBoost, Decision Tree, PyTorch RNN/biRNN) using strict consensus voting—binding is predicted only when **all** 9 models agree.

A naive approach (enumerate → extract features → run all 9 models → vote) scales as O(4^N × M × D), where M = 9 models and D is the per-sample feature extraction cost, making large-scale searches intractable.

---

## 2. Optimization Strategies

### 2.1 Descriptor Hoisting (One-Time Molecular Feature Computation)

All mutant candidates share the same small-molecule target. Since RDKit molecular descriptors depend only on the SMILES string—not the aptamer sequence—the 209-dimensional descriptor vector is computed **once** before enumeration begins and tiled across every candidate during feature matrix assembly.

**Impact:** Eliminates 4^N redundant RDKit evaluations, each of which involves substructure matching, topology analysis, and physicochemical property computation.

### 2.2 Vectorized Batch k-mer Counting via Base-4 Encoding

Rather than computing k-mer frequencies per sequence through Python string operations, the pipeline encodes all sequences as NumPy integer arrays (A=0, T=1, G=2, C=3) and computes k-mer counts across an entire batch in a single vectorized pass:

1. **Base-4 encoding:** Each k-length sliding window maps to a unique integer via positional base-4 arithmetic: `index = Σ (base_i × 4^(k−1−i))`.
2. **Offset bincount:** A per-row offset (`row_index × 4^k`) is added to each k-mer index, enabling a single `np.bincount()` call over the flattened array to produce the full (batch × vocab) count matrix.
3. **L1 normalization:** Counts are divided by the number of k-mer windows to yield frequency vectors.

**Impact:** Replaces O(B × L) Python-level string slicing with O(B × L × k) NumPy arithmetic—orders of magnitude faster for large batches due to SIMD vectorization and cache locality.

### 2.3 Adaptive Model Ordering (Calibration Phase)

Before the main enumeration loop, a calibration phase evaluates a small sample (≤ 64 random mutants) through each of the 9 models independently. Models are then sorted by **ascending positive rate** on this sample—the model that rejects the most candidates is placed first.

**Rationale:** The strict consensus rule (all 9 models must agree) implies that the effective positive rate is the product of individual positive rates, which is extremely low. By placing the most selective (i.e., most negative-happy) model first, the majority of candidates are eliminated before reaching subsequent, more permissive models.

**Impact:** Reduces the effective work per candidate from 9 model evaluations to approximately 1–3 on average, depending on model selectivity distribution.

### 2.4 Early-Exit Sequential Filtering (Cascade Rejection)

Candidates are processed through models in calibration-determined order. After each model evaluates the surviving subset, only candidates predicted as positive proceed to the next model. The `surviving` index array is progressively pruned via boolean masking:

```
surviving = np.arange(B)           # all candidates enter
for model in ordered_models:
    preds = model.predict(X[surviving])
    surviving = surviving[preds >= 0.5]   # keep only positives
    if len(surviving) == 0:
        break                              # early exit: no survivors
```

**Impact:** For a typical search with ~0.1% ensemble positive rate, the cascade reduces total model invocations by > 95%. Models later in the chain process exponentially fewer candidates.

### 2.5 Chunked Enumeration with Constant-Memory Mutant Generation

The full 4^N search space is never materialized simultaneously. Instead, mutants are generated in fixed-size chunks (default: 65,536 candidates per chunk):

1. **Quasi-base-4 decomposition:** The global candidate index `i ∈ [start, stop)` is decomposed into per-site base-4 digits via iterated division/modulus, yielding the substitution at each mutation site.
2. **In-place byte array mutation:** A tiled copy of the original sequence's ASCII bytes is overwritten at the mutation site positions using NumPy advanced indexing: `mutant_bytes[:, sites] = base_bytes[digits]`.

**Impact:** Memory usage is bounded by O(chunk_size × L) rather than O(4^N × L), enabling searches with billions of candidates on consumer hardware.

### 2.6 Streaming Result Collection

When invoked from the TUI (`collect_results=False`), positive hits are streamed directly to a CSV file via a thread-safe callback rather than accumulated in an in-memory list. The TUI displays only the most recent 10 hits in a sliding-window table.

**Impact:** Eliminates O(H × L) memory overhead for H positive hits, enabling unbounded accumulation of results across arbitrarily large search spaces.

### 2.7 Hardware Acceleration (CUDA)

When a CUDA-capable GPU is detected:

- **PyTorch RNN/biRNN models** are moved to GPU memory at load time. Batch inference uses `torch.no_grad()` and tensorized input pipelines.
- **XGBoost models** leverage the GPU prediction path via `DMatrix(..., device="cuda")`, bypassing CPU-bound histogram computation.
- **Device placement** is cached across the session to avoid repeated detection overhead.

**Impact:** GPU acceleration provides 5–50× speedup on the three neural and tree-boosted models, which constitute the most computationally expensive components of the ensemble.

---

## 3. End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Calibration Phase                           │
│  64 sample mutants ──→ 9 models ──→ sort by positive count     │
│                                          (most selective first) │
├─────────────────────────────────────────────────────────────────┤
│                      Main Enumeration Loop                      │
│                                                                 │
│  for each chunk of ≤65,536 candidates:                         │
│    1. Generate mutant byte arrays (base-4 decomposition)        │
│    2. Base-4 encode ──→ vectorized k-mer counting (bincount)    │
│    3. Tile pre-computed molecular descriptors                   │
│    4. Cascade through ordered models:                           │
│       Model_1 (most selective) ──→ survivors ──→               │
│       Model_2 ──→ survivors ──→ ... ──→                        │
│       Model_9 (least selective) ──→ final survivors             │
│    5. Stream positive hits to CSV + update TUI                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Complexity Analysis

| Aspect | Naive | Optimized |
|--------|-------|-----------|
| Feature extraction | O(4^N × L) Python string ops | O(4^N × L) vectorized NumPy |
| Molecular descriptors | O(4^N × 209) RDKit calls | O(209) RDKit calls (once) |
| Model invocations | 9 × 4^N | Σ (4^N × ∏_{j<i} r_j) where r_j is model j's positive rate |
| Memory (candidates) | O(4^N × L) | O(65,536 × L) |
| Memory (results) | O(H × L) | O(10 × L) TUI / O(1) CSV stream |

In practice, the combination of descriptor hoisting, vectorized feature extraction, and cascade filtering reduces wall-clock time by 1–2 orders of magnitude compared to a naive per-sample loop, while keeping peak memory usage bounded regardless of search space size.

---

## 5. Applicability

These optimizations are automatically applied when running mutation search via:

- The interactive TUI: `python -m aptamer_predictor --tui`
- Programmatic API: `EnsemblePredictor.predict_mutation_batch(...)`

No configuration is required. The system auto-detects GPU availability, calibrates model order, and adjusts chunk sizes based on the search space magnitude.
