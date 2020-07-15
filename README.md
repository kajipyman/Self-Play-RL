# Self-Play RL
This is a chainerrl implementation of Self-Play RL.

Self-Play RL is a model in which the agent plays against its copy and receives a reward based on victory or defeat.

We've dealt with tasks of CartPole-v0 and Pendulum-v0 in OpenAI Gym.

## Requirements
* Python 3.7+
* Chainer 7.0+
* ChainerRL 0.8+
* gym 0.17+

## Usage
Train a task of CartPole-v0 with a default reward.
```
python train_cartpole.py
```

Train a task of Pendulum-v0 with a default reward.
```
python train_pendulum.py
```

Train a task of CartPole-v0 with a self-play-based reward.
```
python train_cartpole_selfplay.py
```

Train a task of Pendulum-v0 with a self-play-based reward.
```
python train_pendulum_selfplay.py
```
