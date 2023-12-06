# Reinforcement learning
V této lekci si ukážeme, jak můžeme využít [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
spolu s neuronovými sítěmi pro řešení problémů, ve kterých neznáme správnou odpověď, a nemůžeme tedy
použít klasické učení s učitelem (supervised learning).

## Instalace závislosti
```bash
$ pip install tensorflow gymnasium[classic-control]
```

## Soubory
- [cartpole](cartpole.py) - agent pro řešení problému [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
naimplementovaný pomocí Deep Q-learningu.
