from time import perf_counter

import matplotlib.pyplot as plt
from IPython.display import HTML, display

from mudgym.notebooks import show_episode_transcript


class Screen:
    def __init__(self, lines=16, title=None):
        self.lines = lines
        self.title = title
        self.moves = []
        self.handle = display(HTML(""), display_id=True)
        self.drawn_at = 0.0

    def show(self, transition):
        if transition["step"] == 1:
            self.moves = []
        self.moves.append(transition)
        if perf_counter() - self.drawn_at > 0.1:
            self.draw()

    def draw(self):
        self.handle.update(show_episode_transcript(self.moves, lines=self.lines, title=self.title))
        self.drawn_at = perf_counter()


class LearningCurves:
    def __init__(self, baseline_successes, eval_episodes):
        self.baseline_successes = baseline_successes
        self.eval_episodes = eval_episodes
        self.figure, self.axes = plt.subplots(1, 3, figsize=(15, 3), constrained_layout=True)
        # closed so the cell doesn't show it again when it finishes
        plt.close(self.figure)
        self.handle = display(HTML(""), display_id=True)
        # just the untrained check until the first one comes in
        self.draw([], [], [])

    def draw(self, returns, losses, eval_rows):
        axes = self.axes
        for axis in axes:
            axis.clear()
        episodes = range(1, len(returns) + 1)
        axes[0].plot(episodes, returns)
        axes[0].set(xlabel="Training episode", ylabel="Return", title="Training episodes")
        axes[1].plot(episodes, losses)
        axes[1].set(xlabel="Training episode", ylabel="Mean loss", title="Loss")
        axes[2].plot(
            [0] + [row["After episode"] for row in eval_rows],
            [self.baseline_successes] + [row["Goals reached"] for row in eval_rows],
            marker="o",
        )
        axes[2].set(
            xlabel="Episode",
            ylabel="Goals reached",
            title="Eval",
            ylim=(-0.1, self.eval_episodes + 0.1),
        )
        self.handle.update(self.figure)
