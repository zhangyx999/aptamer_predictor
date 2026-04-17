"""Mutation screen — select mutation sites interactively."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, Label, SelectionList

class MutationScreen(Screen):
    """Screen for selecting which positions to mutate."""

    def __init__(self, sequence: str, **kwargs):
        super().__init__(**kwargs)
        self.sequence = sequence

    def compose(self) -> ComposeResult:
        with Container(id="mutation-container"):
            yield Label("Select Mutation Sites", classes="title")
            yield Label("Space to toggle, Enter or button to confirm")
            yield SelectionList(
                *[
                    (f"{i} ({self.sequence[i]})", i)
                    for i in range(len(self.sequence))
                ],
                id="site-list",
            )
            yield Label("Selected: 0 sites | Space: 4^0 = 1", id="site-counter")
            with Container(classes="button-row"):
                yield Button("Start Prediction", variant="primary", id="predict-btn")
                yield Button("Back", variant="default", id="back-btn")

    def on_selection_list_selected_changed(
        self, event: SelectionList.SelectedChanged
    ) -> None:
        sites = event.selection_list.selected
        n = len(sites)
        space = 4 ** n
        counter = self.query_one("#site-counter", Label)
        counter.update(f"Selected: {n} sites | Space: 4^{n} = {space:,}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
            return

        if event.button.id != "predict-btn":
            return

        site_list = self.query_one("#site-list", SelectionList)
        sites = sorted(site_list.selected)

        if not sites:
            self.query_one("#site-counter", Label).update(
                "Please select at least one site"
            )
            return

        self.app.selected_sites = sites

        from .results_screen import ResultsScreen
        self.app.push_screen(ResultsScreen())
