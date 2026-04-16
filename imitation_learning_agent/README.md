# Imitation Learning Agent

**Agent name:** ImitationLearningAgent

**Author(s):** Bo Feng (bfeng66@gatech.edu), Frank Yang (fyang338@gatech.edu)

## Description

A PPO agent trained with **Behavioral Cloning (BC) + PPO fine-tuning**:

1. **Expert data collection**: Collected ~1M (observation, action) pairs from our best trained agent (PPO + Reward Shaping + Self-Play, 80% win rate vs CEIA baseline)
2. **BC pretraining**: Trained a policy network via supervised learning (cross-entropy loss) on expert demonstrations for 50 epochs
3. **PPO fine-tuning**: Injected BC-pretrained weights into a PPO trainer and fine-tuned against 100% CEIA baseline opponent with optimized hyperparameters (low learning rate to preserve BC initialization, higher entropy coefficient to encourage exploration)

This approach addresses the cold-start problem: training from scratch against CEIA produces no learning (reward ~0.078), but BC pretraining provides a strong initialization that enables effective fine-tuning.
