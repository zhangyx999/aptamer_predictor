"""Model loading, prediction, and ensemble voting."""

from __future__ import annotations

import glob
import os
import pickle
import re
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# PyTorch RNN wrapper  (loaded lazily to avoid hard import)
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = False
_nn = None


def _ensure_torch():
    global _TORCH_AVAILABLE, _nn
    if _nn is None:
        try:
            import torch
            import torch.nn as nn
            _nn = nn
            _TORCH_AVAILABLE = True
        except ImportError:
            raise ImportError(
                "PyTorch is required for loading RNN/biRNN models. "
                "Install it with: pip install torch"
            )


class SimpleRNN:
    """PyTorch RNN model wrapper compatible with sklearn-style API.

    Must be defined in __main__ namespace before pickle.load() so that
    the deserializer can find the class.
    """

    # Will be set to the actual nn.Module subclass after _ensure_torch()
    _nn_module = None

    @classmethod
    def _as_module(cls):
        """Return a proper nn.Module subclass that pickle can instantiate."""
        if cls._nn_module is not None:
            return cls._nn_module

        _ensure_torch()
        import torch
        import torch.nn as nn

        class _SimpleRNN(nn.Module):
            """RNN -> Linear -> Sigmoid -> Linear -> Sigmoid."""

            def forward(self, x):
                if not isinstance(x, torch.Tensor):
                    x = torch.FloatTensor(np.asarray(x, dtype=np.float32))
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                # RNN expects (seq_len, batch, input_size)
                x = x.unsqueeze(0)  # (1, batch, features)
                out, _ = self.rnn(x)
                out = out.squeeze(0)  # (batch, hidden * num_directions)
                out = self.sig1(self.fc1(out))
                out = self.sig2(self.fc2(out))  # (batch, 1)
                return out.squeeze(-1)  # (batch,)

            def predict_proba(self, X):
                self.eval()
                with torch.no_grad():
                    if not isinstance(X, np.ndarray):
                        X = np.asarray(X, dtype=np.float32)
                    t = torch.FloatTensor(X)
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    out = self.forward(t).cpu().numpy()
                    return np.column_stack([1 - out, out])

            def predict(self, X):
                probs = self.predict_proba(X)
                return (probs[:, 1] >= 0.5).astype(int)

        cls._nn_module = _SimpleRNN
        return _SimpleRNN


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _extract_mer_label(filename: str) -> Optional[str]:
    """Extract mer label like '1mer', '23mer' from a model filename."""
    m = re.search(r"\((\d+mer)\)", filename)
    return m.group(1) if m else None


def load_model(filepath: str):
    """Load a single .pkl model, handling both sklearn and PyTorch types."""
    mer = _extract_mer_label(os.path.basename(filepath))
    basename = os.path.basename(filepath)

    # Check if this is a PyTorch model by inspecting filename
    is_pytorch = basename.endswith("RNN.pkl") or basename.endswith("biRNN.pkl")

    if is_pytorch:
        _ensure_torch()
        import __main__
        __main__.SimpleRNN = SimpleRNN._as_module()
        with open(filepath, "rb") as f:
            model = pickle.load(f)
    else:
        with open(filepath, "rb") as f:
            model = pickle.load(f)

    return model, mer


# ---------------------------------------------------------------------------
# Ensemble predictor
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    """Manage all 9 pre-trained models and perform ensemble prediction."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = []  # list of (model, mer_label, filepath)
        self._load_all()

    def _load_all(self):
        pattern = os.path.join(self.model_dir, "(*mer)*.pkl")
        files = sorted(glob.glob(pattern))
        if not files:
            # Also try without parens pattern for directories with spaces
            pattern = os.path.join(self.model_dir, "*.pkl")
            files = sorted(glob.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No .pkl model files found in {self.model_dir}"
            )

        for fp in files:
            try:
                model, mer = load_model(fp)
                self.models.append((model, mer, os.path.basename(fp)))
                print(f"  Loaded: {os.path.basename(fp)}")
            except Exception as e:
                print(f"  Warning: failed to load {os.path.basename(fp)}: {e}")

    # ---- single sample ---------------------------------------------------

    def predict_one(self, sequence: str, smiles: str) -> dict:
        """Predict with all models on a single aptamer-SMILES pair.

        Args:
            sequence: Aptamer sequence (RNA or DNA).
            smiles: Small molecule SMILES string.

        Returns:
            dict with keys: individual (dict of model->pred), ensemble_label
        """
        from aptamer_predictor.features import MER_K_MAP, build_feature_vector

        results = {}
        model_labels = []

        for model, mer, fname in self.models:
            if mer is None or mer not in MER_K_MAP:
                continue
            feat = build_feature_vector(sequence, smiles, MER_K_MAP[mer])
            pred = model.predict(feat.reshape(1, -1))[0]
            prob = model.predict_proba(feat.reshape(1, -1))[0, 1]
            results[fname] = {"label": int(pred), "probability": float(prob)}
            model_labels.append(int(pred))

        # Ensemble label: 1 only if all 9 models predict 1, otherwise 0
        ensemble_label = 1 if all(label == 1 for label in model_labels) else 0

        return {
            "individual": results,
            "ensemble_label": ensemble_label,
        }

    # ---- batch prediction with raw aptamer + SMILES ----------------------

    def predict_batch(
        self,
        sequences: list[str],
        smiles_list: list[str],
        labels: Optional[list[int]] = None,
        ids: Optional[list] = None,
    ) -> list[dict]:
        """Full pipeline: extract features → predict with all models → ensemble vote.

        Args:
            sequences: List of aptamer sequences (RNA or DNA).
            smiles_list: List of SMILES strings.
            labels: Optional true labels for evaluation.
            ids: Optional sample IDs.

        Returns:
            List of result dicts.
        """
        from aptamer_predictor.features import MER_K_MAP, build_feature_vector

        all_results = []

        for i, (seq, smi) in enumerate(zip(sequences, smiles_list)):
            sample = {
                "sequence": seq,
                "smiles": smi,
            }
            if ids is not None:
                sample["id"] = ids[i]
            if labels is not None:
                sample["true_label"] = labels[i]

            individual = {}
            model_labels = []

            for model, mer, fname in self.models:
                if mer is None or mer not in MER_K_MAP:
                    continue
                k_list = MER_K_MAP[mer]
                feat = build_feature_vector(seq, smi, k_list)

                pred = model.predict(feat.reshape(1, -1))[0]
                prob = model.predict_proba(feat.reshape(1, -1))[0, 1]

                individual[fname] = {
                    "label": int(pred),
                    "probability": round(float(prob), 6),
                }
                model_labels.append(int(pred))

            # Ensemble label: 1 only if all 9 models predict 1, otherwise 0
            ensemble_label = 1 if all(label == 1 for label in model_labels) else 0

            sample["individual"] = individual
            sample["ensemble_label"] = ensemble_label
            all_results.append(sample)

            if (i + 1) % 50 == 0 or i == len(sequences) - 1:
                print(f"  Processed {i + 1}/{len(sequences)} samples")

        return all_results

    # ---- evaluate on pre-formatted CSV -----------------------------------

    def evaluate(self, data_dir: str, output_dir: str):
        """Evaluate all models on pre-formatted test CSV files.

        Reads test data files matching pattern:
          {data_dir}/{mer}independent_data all des(104-2080).csv

        Writes per-model predictions and summary metrics to output_dir.
        """
        import csv
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        os.makedirs(output_dir, exist_ok=True)

        def calc_specificity(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)

        metrics_path = os.path.join(output_dir, "all_models_metrics.csv")
        with open(metrics_path, "w", newline="") as mf:
            writer = csv.writer(mf)
            writer.writerow([
                "Model", "AUC", "Accuracy", "F1-score",
                "Precision", "Recall", "Specificity",
            ])

        for model, mer, fname in self.models:
            if mer is None:
                continue

            # Find matching test data file
            pattern = os.path.join(
                data_dir, f"{mer}*independent_data*all*des*.csv"
            )
            data_files = glob.glob(pattern)

            if not data_files:
                print(f"  Skipping {fname}: no test data found for {mer}")
                continue

            data_file = data_files[0]
            print(f"  Evaluating {fname} with {os.path.basename(data_file)}")

            # Read test data
            test_x, label, seq, pubchem_id = [], [], [], []
            with open(data_file, "r") as g:
                g.readline()  # skip header
                for line in g:
                    items = line.strip().split(",")
                    for j in range(3, len(items)):
                        items[j] = float(items[j])
                    test_x.append(items[3:])
                    label.append(items[0])
                    seq.append(items[1])
                    pubchem_id.append(items[2])

            test_x = np.array(test_x)
            label = np.array(label).astype(int)

            # Predict
            pred_y = model.predict(test_x)
            pred_y_prob = model.predict_proba(test_x)[:, 1]

            # Write per-model predictions
            out_name = fname.replace(".pkl", "") + "_predictions.csv"
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, "w") as f:
                f.write("true_label,seq,pubchemid,pred_label,prob\n")
                for u in range(len(seq)):
                    f.write(
                        f"{label[u]},{seq[u]},{pubchem_id[u]},"
                        f"{pred_y[u]},{pred_y_prob[u]}\n"
                    )

            # Compute metrics
            auc = roc_auc_score(label, pred_y_prob)
            accuracy = accuracy_score(label, pred_y)
            precision = precision_score(label, pred_y)
            f1 = f1_score(label, pred_y)
            recall = recall_score(label, pred_y)
            specificity = calc_specificity(label, pred_y)

            with open(metrics_path, "a", newline="") as mf:
                writer = csv.writer(mf)
                writer.writerow([
                    fname.replace(".pkl", ""),
                    f"{auc:.4f}", f"{accuracy:.4f}", f"{f1:.4f}",
                    f"{precision:.4f}", f"{recall:.4f}", f"{specificity:.4f}",
                ])

            print(
                f"    AUC={auc:.4f}  Acc={accuracy:.4f}  "
                f"F1={f1:.4f}  Prec={precision:.4f}  "
                f"Recall={recall:.4f}  Spec={specificity:.4f}"
            )

        print(f"\nMetrics saved to {metrics_path}")
