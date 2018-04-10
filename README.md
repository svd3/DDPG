# Deep Deterministic Policy Gradient (DDPG)

This repository implements a DDPG agent for "Firefly" task (my research project).
It can be used for other gym tasks as well, by replacing environment name in `env.make()` in `main.py`
use `--play` to set play mode to see the trained agent perform, not using that automatically sets it in training mode and updates the agent parameters.
use `--render` to visualize the environment. Note: for different environments you might need to change render function.
