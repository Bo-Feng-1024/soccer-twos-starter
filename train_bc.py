"""
Behavioral Cloning (BC) pretraining for SoccerTwos.

Trains a policy network via supervised learning on expert demonstration data.
The model architecture matches Ray PPO's FullyConnectedNetwork exactly,
so weights can be directly injected into a PPOTrainer for fine-tuning.

Usage:
    python train_bc.py [--data PATH] [--output PATH] [--epochs N]
"""
import argparse
import os

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


DEFAULT_DATA = "expert_data.npz"
DEFAULT_OUTPUT = "bc_weights.pt"
DEFAULT_EPOCHS = 50
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.1


def build_model():
    """Build a FullyConnectedNetwork matching the PPO policy architecture."""
    from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(336,), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([3, 3, 3])
    model_config = {
        "vf_share_layers": True,
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    }
    model = FullyConnectedNetwork(obs_space, act_space, 9, model_config, "bc_model")
    return model


def bc_loss(logits, actions):
    """Cross-entropy loss over 3 sub-actions (MultiDiscrete[3,3,3])."""
    loss = 0.0
    for i in range(3):
        sub_logits = logits[:, i * 3 : (i + 1) * 3]
        loss += F.cross_entropy(sub_logits, actions[:, i])
    return loss / 3.0


def bc_accuracy(logits, actions):
    """Top-1 accuracy averaged over 3 sub-actions."""
    correct = 0
    total = 0
    for i in range(3):
        sub_logits = logits[:, i * 3 : (i + 1) * 3]
        pred = torch.argmax(sub_logits, dim=1)
        correct += (pred == actions[:, i]).sum().item()
        total += actions.size(0)
    return correct / total


def train(data_path, output_path, epochs):
    """Train BC model on expert data."""
    # Load data
    print(f"Loading data from {data_path}")
    data = np.load(data_path)
    observations = torch.tensor(data["observations"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.long)
    print(f"  Samples: {len(observations):,}")
    print(f"  Observation shape: {observations.shape}")
    print(f"  Action shape: {actions.shape}")

    # Train/val split
    dataset = TensorDataset(observations, actions)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Train: {train_size:,}, Val: {val_size:,}")

    # Build model
    model = build_model()
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for obs_batch, act_batch in train_loader:
            logits, _ = model({"obs_flat": obs_batch, "obs": obs_batch}, [], None)
            loss = bc_loss(logits, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                logits, _ = model({"obs_flat": obs_batch, "obs": obs_batch}, [], None)
                val_loss_sum += bc_loss(logits, act_batch).item()
                val_acc_sum += bc_accuracy(logits, act_batch)
                val_batches += 1

        train_loss = train_loss_sum / train_batches
        val_loss = val_loss_sum / val_batches
        val_acc = val_acc_sum / val_batches
        lr = scheduler.get_last_lr()[0]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.3f} | "
                f"lr={lr:.6f}{marker}"
            )

    # Save best model
    torch.save(best_state, output_path)
    print(f"\nSaved best model (val_loss={best_val_loss:.4f}) to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC pretraining for SoccerTwos")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to expert_data.npz")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output weights path")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    args = parser.parse_args()

    train(args.data, args.output, args.epochs)
