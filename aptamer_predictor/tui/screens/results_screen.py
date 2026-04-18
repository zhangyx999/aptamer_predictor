"""Results screen — progress bar + streaming results table + auto CSV export."""

from __future__ import annotations

import csv
import threading
import time
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
        self._csv_file = None
        self._csv_writer = None
        self._csv_lock = threading.Lock()
        self._csv_filename: str = ""
        self._hit_count = 0
        self._start_time = 0.0

    def compose(self) -> ComposeResult:
        with Container(id="results-container"):
            yield Label("Prediction Results", classes="title")
            yield ProgressBar(total=100, id="progress-bar")
            yield Label("Preparing...", id="progress-label")
            yield Label("Hits: 0", id="hit-counter")
            yield DataTable(id="results-table")
            with Container(classes="button-row"):
                yield Button("New Search", variant="default", id="new-btn")

    def on_mount(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.add_columns("Rank", "Sequence", "Prob")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        requested_filename = getattr(self.app, "result_filename", "").strip()
        self._csv_filename = requested_filename or f"{ts}.csv"
        self._csv_file = open(self._csv_filename, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "sequence",
            "mean_probability",
        ])
        self._csv_file.flush()

        app = self.app
        assert app.predictor is not None
        total_mutations = 4 ** len(app.selected_sites)
        self._start_time = time.monotonic()
        self._update_progress(0, total_mutations, 0.0)

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

        total_mutations = 4 ** len(sites)
        batch_size = max(50, total_mutations // 200)

        def on_result(result: dict) -> None:
            with self._csv_lock:
                if self._csv_writer is None or self._csv_file is None:
                    return
                self._csv_writer.writerow([
                    result["sequence"],
                    result["mean_probability"],
                ])
                self._csv_file.flush()
            self._hit_count += 1
            self.app.call_from_thread(self._update_hit_counter)

        def on_progress(done: int, total: int, info: dict) -> None:
            pct = (done / total * 100) if total > 0 else 0
            self.app.call_from_thread(self._update_progress, done, total, pct)

        try:
            predictor.predict_mutation_batch(
                sequence,
                smiles,
                sites,
                batch_size=batch_size,
                progress_callback=on_progress,
                should_cancel=lambda: self._should_cancel(worker),
                result_callback=on_result,
                collect_results=False,
            )
        except PredictionCancelled:
            self.app.call_from_thread(self._handle_prediction_cancelled)
            return
        except Exception as exc:
            self.app.call_from_thread(self._show_error, str(exc))
            return
        finally:
            self._close_csv()

        results = self._load_results_from_csv()
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
        self.query_one("#new-btn", Button).disabled = True

        worker = self._prediction_worker
        if worker is not None:
            cancel = getattr(worker, "cancel", None)
            if callable(cancel):
                cancel()

    def _return_to_input(self) -> None:
        self.app.pop_screen()
        self.app.pop_screen()

    def _set_label(self, text: str) -> None:
        if self._cancel_event.is_set() or self._is_unmounted:
            return
        self.query_one("#progress-label", Label).update(text)

    def _format_speed(self, speed: float) -> str:
        if speed >= 1_000_000:
            return f"{speed / 1_000_000:.1f}M/s"
        if speed >= 1_000:
            return f"{speed / 1_000:.1f}K/s"
        return f"{speed:.0f}/s"

    def _update_progress(self, done: int, total: int, pct: float) -> None:
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.update(progress=pct)
        elapsed = time.monotonic() - self._start_time if self._start_time else 1.0
        speed = done / elapsed if elapsed > 0 else 0.0
        self._set_label(
            f"{done:,} / {total:,} ({pct:.1f}%) | {self._format_speed(speed)} → {self._csv_filename}"
        )

    def _update_hit_counter(self) -> None:
        if self._cancel_event.is_set() or self._is_unmounted:
            return
        self.query_one("#hit-counter", Label).update(
            f"Hits: {self._hit_count:,}"
        )

    def _load_results_from_csv(self) -> list[dict]:
        if not self._csv_filename:
            return []

        results: list[dict] = []
        with open(self._csv_filename, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                results.append(
                    {
                        "sequence": row["sequence"],
                        "mean_probability": float(row["mean_probability"]),
                    }
                )

        results.sort(key=lambda r: r["mean_probability"], reverse=True)
        return results

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
                f"{result['mean_probability']:.4f}",
            )

        label.update(
            f"Done — {len(results)} candidates → {self._csv_filename}"
        )
        self.query_one("#new-btn", Button).disabled = False

    def _show_error(self, message: str) -> None:
        self._prediction_done = True
        self.query_one("#progress-label", Label).update(f"Error: {message}")
        self.query_one("#new-btn", Button).disabled = False

    def _handle_prediction_cancelled(self) -> None:
        self._prediction_done = True
        if self._is_unmounted:
            return

        self.query_one("#progress-label", Label).update("Prediction cancelled")

        if self._return_to_input_pending:
            self._return_to_input()
            return

        self.query_one("#new-btn", Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-btn":
            if self._prediction_done:
                self._return_to_input()
                return

            self._return_to_input_pending = True
            self._cancel_prediction()

    def _close_csv(self) -> None:
        if self._csv_file is not None:
            with self._csv_lock:
                self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def on_unmount(self) -> None:
        self._is_unmounted = True
        if not self._prediction_done:
            self._cancel_event.set()
            worker = self._prediction_worker
            if worker is not None:
                cancel = getattr(worker, "cancel", None)
                if callable(cancel):
                    cancel()
