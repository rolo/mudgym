# Notebooks

Run these notebooks in your browser with JupyterLite. Each notebook opens in a new tab, with no installation needed.

The source notebooks and their helper modules live in `docs/notebooks/` in the repository. The downloads below are the copies used by JupyterLite, with saved outputs cleared.

## Quickstart

Create a MudGym environment, inspect its observations, and send a text action. View the game output and the reward returned by the environment.

Repository path: `docs/notebooks/quickstart.ipynb`.

<a href="../lite/notebooks/index.html?path=quickstart.ipynb" target="_blank" rel="noopener">Open the Quickstart notebook</a> or <a href="../lite/files/quickstart.ipynb" download>download the notebook</a>.

## Goal room DQN

Train a deep Q-network to reach the Woodsman's Hut. Follow training progress, inspect evaluation results, and view the agent's routes and game transcripts.

Repository path: `docs/notebooks/goal_room_dqn.ipynb`.

<a href="../lite/notebooks/index.html?path=goal_room_dqn.ipynb" target="_blank" rel="noopener">Open the Goal room DQN notebook</a> or <a href="../lite/files/goal_room_dqn.ipynb" download>download the notebook</a>.

If you download this notebook, also save <a href="../lite/files/display.py" download>display.py</a> and <a href="../lite/files/numpy_torch.py" download>numpy_torch.py</a> in the same folder. These helper modules are already included in a repository checkout.

## Run locally

From the repository root, run:

```sh
just lab
```

This opens `docs/notebooks/` in JupyterLab with the notebook dependencies installed.
