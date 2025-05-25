# Robot Soccer Simulation (Leiden University Robotics 2025)

This project is developed as the final project for the Robotics 2025 course at LIACS, Leiden University.

## Overview

This project extends a 2D robot football simulation environment based on the open-source repository [`robot_soccer_python`](https://github.com/jonysalgado/robot_soccer_python). It integrates both manual control logic and several reinforcement learning algorithms for competitive 2v2 robot soccer.

## Implemented Controllers

- Manual rule-based control
- Deep Q-Network (DQN)
- Double DQN
- Dueling DQN
- Discrete Soft Actor-Critic (SAC)

## Code Structure

- `robot_soccer_python/` — 2D simulation environment (based on open-source project)
- `Handrules.py` — Manual rule-based controller for 2v2 games
- `DQN_VS_DDQN.py` — DQN vs. Double DQN controller for 2v2 games
- `DQN_VS_DuelingDQN.py` — DQN vs. Dueling DQN controller for 2v2 games
- `DQN_VS_SAC.py` — DQN vs. Discrete SAC controller for 2v2 games
- `plot_control_heatmap.py` — Generates pie charts of ball possession after matches

## Video Demo

Watch the gameplay demo here: [https://youtu.be/ufvMlBDDpaw](https://youtu.be/ufvMlBDDpaw)


## How to Run

Install the required dependencies:

```bash
pip install robot_soccer_python
pip install torch numpy matplotlib pygame
```

Then run the desired matchup:

```bash
# Manual rule-based controller
python Handrules.py

# DQN vs Double DQN
python DQN_VS_DDQN.py

# DQN vs Dueling DQN
python DQN_VS_DuelingDQN.py

# DQN vs Discrete SAC
python DQN_VS_SAC.py
```

After training or evaluation, use the following script to visualize ball possession:

```bash
python plot_control_heatmap.py
```

Ensure that the simulation starts with visual display enabled and log files will be generated automatically for analysis.
