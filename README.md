# Soccer-Twos Starter Kit

Example training/testing scripts for the Soccer-Twos environment. This starter code is modified from the example code provided in https://github.com/bryanoliveira/soccer-twos-starter.

Environment-level specification code can be found at https://github.com/bryanoliveira/soccer-twos-env, which may also be useful to reference.

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



