# Soccer-Twos — CS8803 DRL Final Project (Spring 2026)

🏆 **1st place / 72 teams** — Georgia Tech CS8803 Deep Reinforcement Learning

This repository contains our final project submission for CS8803 Deep Reinforcement Learning. The starter code is forked from [bryanoliveira/soccer-twos-starter](https://github.com/bryanoliveira/soccer-twos-starter); environment specification lives at [bryanoliveira/soccer-twos-env](https://github.com/bryanoliveira/soccer-twos-env).

## Results

| Metric                                       | Result            |
| ---                                          | ---               |
| Final Project Tournament (72 teams)          | 🥇 **1st place**  |
| vs. CEIA baseline (100 matches, headless)    | **90/100 = 90%**  |
| vs. Random opponent (100 matches, headless)  | **99/100 = 99%**  |

## Authors

- Bo Feng (bfeng66@gatech.edu)
- Frank F. Yang (frank.yang@gatech.edu)

## Report

- Compiled PDF: [`report/CS8803_DRL_Final_Report.pdf`](report/CS8803_DRL_Final_Report.pdf)
- LaTeX source: [`report/report.tex`](report/report.tex)

## Submitted Agents

We trained and analysed three agents, all PPO-based via Ray RLlib:

| Folder | Description |
| --- | --- |
| `reward_shaping_ppo_agent/` | **Submitted agent (Agent 2)** — PPO + reward shaping + 70/30 self-play/CEIA mix with separated policy/value networks; the 1st-place agent |
| `Answer to the Ultimate Question of Life, The Universe, and Everything_AGENT/` | Same agent, packaged as the tournament submission zip |
| `imitation_learning_agent/` | Bonus Agent 3 — BC pretrain + PPO fine-tune (negative result, analysed in report §4) |
| `ceia_baseline_agent/` | Course-provided baseline (used as evaluation opponent and as 30% of the opponent pool) |
| `example_player_agent/` | TA's player-level agent template (unmodified) |
| `example_team_agent/` | TA's team-level agent template (unmodified) |

## Reward Modification

The reward-shaping wrapper (the rubric's environment-modification component, +40 pts) is implemented in [`utils.py`](utils.py), lines 18–102 (`RewardShapingWrapper`). The same file is also bundled inside the submitted agent zip at [`Answer to the Ultimate Question of Life, The Universe, and Everything_AGENT/utils.py`](Answer%20to%20the%20Ultimate%20Question%20of%20Life%2C%20The%20Universe%2C%20and%20Everything_AGENT/utils.py). It adds six dense per-step signals on top of the sparse goal reward:

| Signal       | Weight    | Description                                            |
| ---          | ---       | ---                                                    |
| Approach     | +0.0002   | Velocity component toward the ball                     |
| Kick         | +0.001    | Ball acceleration when within 1.5 units of the ball    |
| Offensive    | +0.0004   | Ball velocity toward the opponent goal                 |
| Defensive    | -0.001    | Ball within 5 units of own goal                        |
| Time penalty | -0.00002  | Per-step cost to discourage stalling                   |
| Separation   | +0.0001   | Distance between the two teammates                     |

All weights are kept small (max |w| = 1e-3) so the original ±1 goal reward still dominates as the policy improves; this follows the principle that auxiliary shaping should not overwhelm the true return (Ng et al. 1999).

## Quick Evaluation

After completing the install steps below, the submitted agent can be evaluated end-to-end:

```bash
# Watch a single match (visual)
python -m soccer_twos.watch -m1 reward_shaping_ppo_agent -m2 ceia_baseline_agent

# Headless: 100 matches vs. CEIA baseline
python scripts/eval_vs_ceia.py 100

# Headless: 100 matches vs. random opponent
python scripts/eval_vs_random.py 100
```

The 99/100 vs. Random run log is at [`report/eval_vs_random_pace.log`](report/eval_vs_random_pace.log); the 90/100 vs. CEIA run is recorded in `notes/training-log.md` (Job #28, checkpoint-2300).

## Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)

## Usage

### 1. Fork this repository

```bash
git clone https://github.com/your-github-user/soccer-twos-starter.git
cd soccer-twos-starter/
```

### 2. Create and activate conda environment

```bash
conda create --name soccertwos python=3.8 -y
conda activate soccertwos
```

### 3. Downgrade build tools for compatibility

```bash
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge
```

### 4. Install dependencies

> **Note:** `requirements.txt` specifies `ray==1.4.0` and `torch<1.9.0`, which are no longer available (especially on macOS Apple Silicon). Install in stages instead:

```bash
# mlagents dependencies
pip install grpcio Pillow pyyaml cloudpickle h5py tensorboard numpy

# mlagents packages — use --no-deps to bypass the outdated torch<1.9.0 constraint
pip install mlagents-envs==0.27.0 gym-unity==0.27.0 mlagents==0.27.0 --no-deps
pip install "gym==0.19.0" --no-deps
pip install soccer-twos --no-deps

# PyTorch (arm64 macOS requires >= 1.9.0)
pip install torch

# Remaining dependencies
pip install aiohttp==3.7.4 aioredis==1.3.1 dm-tree==0.1.6 "cattrs>=1.1.0,<1.7"

# Ray — 1.4.0 is no longer published; 1.13.0 is the last 1.x release with compatible API
pip install ray==1.13.0 --no-deps
pip install "click<=8.0.4,>=7.0" "msgpack<2.0.0,>=1.0.0" jsonschema aiosignal frozenlist virtualenv pandas tabulate tensorboardX lz4 matplotlib scikit-image scipy
```

### 5. Fix protobuf and pydantic compatibility

```bash
pip install protobuf==3.20.3
pip install pydantic==1.10.13
```

### 6. Apply macOS compatibility patches (macOS only)

Two installed packages need patching on macOS Apple Silicon:
- `mlagents_envs/rpc_utils.py`: uses `np.bool` which was removed in NumPy 1.24
- `soccer_twos/package.py`: passes the full binary path to mlagents, but mlagents expects only the base path on macOS (it appends `.app/Contents/MacOS/...` itself)

```bash
python scripts/mac_patches.py
```

### 7. Run `python example_random_players.py` to watch random agents play

```bash
python example_random_players.py
```

### 8. Train using any of the example scripts

```bash
python example_ray_ppo_sp_still.py
python example_ray_team_vs_random.py
# etc.
```

## Agent Packaging

To receive full credit on the assignment and ensure the teaching staff can properly compile your code, you must follow these instructions:

- Implement a class that inherits from `soccer_twos.AgentInterface` and implements an `act` method. Examples are located under the `example_player_agent/` or `example_team_agent/` directories.
- Fill in your agent's information in the `README.md` file (agent name, authors & emails, and description)
- Compress each agent's module folder as `.zip`.

*Submission Policy*: Students must submit multiple trained agents to meet all assignment requirements. In both the agent desription and the report, clearly identify which agent file corresponds to each evaluation criterion (e.g., Agent1 – policy performance, Agent2 – reward modification, Agent3 – imitation learning, etc.). 

Training plots are required for every agent that is discussed or submitted. Additionally, include a direct performance comparison across agents, such as overlaid learning curves, to support your analysis.


## Testing/Evaluating

Use the environment's rollout tool to test the example agent module:

`python -m soccer_twos.watch -m example_player_agent`

Similarly, you can test your own agent by replacing `example_player_agent` with the name of your agent directory.

The baseline agent is located here: [pre-trained baseline (download)](https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view?usp=sharing).
To examine the baseline agent, you must extract the `ceia_baseline_agent` folder to this project's folder. For instance you can run, 

`python -m soccer_twos.watch -m1 example_player_agent -m2 ceia_baseline_agent`

, to examine the random agent vs. the baseline agent.



