"""Results screen — progress bar + streaming results table + export."""

from __future__ import annotations

import csv
import threading
from datetime import datetime

import textual.worker as textual_worker
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, ProgressBar
from textual.worker import Worker

from aptamer_predictor.predictor import PredictionCancelled


class ResultsScreen(Screen):
    """Screen showing prediction progress and results."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results: list[dict] = []
        self._prediction_worker: Worker | None = None
        self._cancel_event = threading.Event()
        self._prediction_done = False
        self._return_to_input_pending = False
        self._is_unmounted = False

    def compose(self) -> ComposeResult:
        with Container(id="results-container"):
            yield Label("Prediction Results", classes="title")
            yield ProgressBar(total=100, id="progress-bar")
            yield Label("Preparing...", id="progress-label")
            yield DataTable(id="results-table")
            with Container(classes="button-row"):
                yield Button("Export CSV", variant="success", id="export-btn")
                yield Button("New Search", variant="default", id="new-btn")

    def on_mount(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.add_columns("Rank", "Sequence", "Mutations", "Prob", "Label")
        self.query_one("#export-btn", Button).disabled = True

        app = self.app
        assert app.predictor is not None

        self._prediction_worker = self.run_worker(
            self._run_prediction,
            name="prediction",
            thread=True,
        )

    def _run_prediction(self) -> None:
        app = self.app
        predictor = app.predictor
        assert predictor is not None
        get_current_worker = getattr(textual_worker, "get_current_worker", None)
        worker = get_current_worker() if callable(get_current_worker) else None

        sequence = app.sequence
        smiles = app.smiles
        sites = app.selected_sites

        def on_progress(done: int, total: int, info: dict) -> None:
            pct = int(done / total * 100) if total > 0 else 0
            self.app.call_from_thread(self._update_progress, done, total, pct)

        try:
            results = predictor.predict_mutation_batch(
                sequence,
                smiles,
                sites,
                batch_size=1000,
                progress_callback=on_progress,
                should_cancel=lambda: self._should_cancel(worker),
            )
        except PredictionCancelled:
            self.app.call_from_thread(self._handle_prediction_cancelled)
            return
        except Exception as exc:
            self.app.call_from_thread(self._show_error, str(exc))
            return

        self.app.call_from_thread(self._show_results, results)

    def _should_cancel(self, worker: Worker | None) -> bool:
        if self._cancel_event.is_set():
            return True

        if worker is None:
            return False

        is_cancelled = getattr(worker, "is_cancelled", None)
        if callable(is_cancelled):
            return bool(is_cancelled())
        if is_cancelled is not None:
            return bool(is_cancelled)
        return False

    def _cancel_prediction(self) -> None:
        self._cancel_event.set()
        self.query_one("#progress-label", Label).update("Cancelling...")
        self.query_one("#export-btn", Button).disabled = True
        self.query_one("#new-btn", Button).disabled = True

        worker = self._prediction_worker
        if worker is not None:
            cancel = getattr(worker, "cancel", None)
            if callable(cancel):
                cancel()

    def _return_to_input(self) -> None:
        self.app.pop_screen()
        self.app.pop_screen()

    def _update_progress(self, done: int, total: int, pct: int) -> None:
        if self._cancel_event.is_set() or self._is_unmounted:
            return

        bar = self.query_one("#progress-bar", ProgressBar)
        label = self.query_one("#progress-label", Label)
        bar.update(progress=pct)
        label.update(f"{done:,} / {total:,} ({pct}%)")

    def _show_results(self, results: list[dict]) -> None:
        if self._cancel_event.is_set():
            self._handle_prediction_cancelled()
            return

        self._prediction_done = True
        self._results = results
        table = self.query_one("#results-table", DataTable)
        label = self.query_one("#progress-label", Label)

        for rank, result in enumerate(results, 1):
            seq_short = (
                result["sequence"][:30] + "..."
                if len(result["sequence"]) > 30
                else result["sequence"]
            )
            table.add_row(
                str(rank),
                seq_short,
                result["mutations"],
                f"{result['mean_probability']:.4f}",
                "Bind",
            )

        label.update(f"Done — {len(results)} positive candidates (all 9 models agree)")
        self.query_one("#export-btn", Button).disabled = not bool(results)
        self.query_one("#new-btn", Button).disabled = False

    def _show_error(self, message: str) -> None:
        self._prediction_done = True
        self.query_one("#progress-label", Label).update(f"Error: {message}")
        self.query_one("#export-btn", Button).disabled = True
        self.query_one("#new-btn", Button).disabled = False

    def _handle_prediction_cancelled(self) -> None:
        self._prediction_done = True
        if self._is_unmounted:
            return

        self.query_one("#progress-label", Label).update("Prediction cancelled")
        self.query_one("#export-btn", Button).disabled = True

        if self._return_to_input_pending:
            self._return_to_input()
            return

        self.query_one("#new-btn", Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "export-btn":
            self._export_csv()
        elif event.button.id == "new-btn":
            if self._prediction_done:
                self._return_to_input()
                return

            self._return_to_input_pending = True
            self._cancel_prediction()

    def on_unmount(self) -> None:
        self._is_unmounted = True
        if not self._prediction_done:
            self._cancel_event.set()
            worker = self._prediction_worker
            if worker is not None:
                cancel = getattr(worker, "cancel", None)
                if callable(cancel):
                    cancel()

    def _export_csv(self) -> None:
        if not self._results:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mutation_predictions_{ts}.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "sequence", "mutations",
                "mean_probability", "ensemble_label",
            ])
            for rank, result in enumerate(self._results, 1):
                writer.writerow([
                    rank,
                    result["sequence"],
                    result["mutations"],
                    result["mean_probability"],
                    result["ensemble_label"],
                ])

        self.query_one("#progress-label", Label).update(
            f"Exported to {filename}"
        )
