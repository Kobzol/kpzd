import collections
import dataclasses
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.random
import tensorflow as tf
from tqdm import tqdm

import gymnasium as gym

random.seed(42)
numpy.random.seed(42)
tf.random.set_seed(42)


@dataclasses.dataclass(frozen=True)
class StepHistoryItem:
    action: int
    state: np.ndarray
    next_state: Optional[np.ndarray]

    def is_terminal(self) -> bool:
        return self.next_state is None

    def reward(self) -> float:
        return 1.0 if self.next_state is not None else 0.0


def make_sample(state: np.ndarray, action: int) -> np.ndarray:
    return np.array(list(state) + [action], dtype=np.float32)


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=100000)
        self.random = random.Random(42)

    def push(self, item: StepHistoryItem):
        self.buffer.append(item)

    def generate_batch(self, size: int, predict_net: "DQN", discount_rate: float) -> Tuple[
        np.ndarray, np.ndarray]:
        items = self.random.choices(self.buffer, k=size)
        batch_x = []
        batch_y = []
        for item in items:
            if item.is_terminal():
                target_q = 0
            else:
                target_q = 1 + discount_rate * np.max(
                    predict_net.predict_q(item.next_state))
            batch_y.append(target_q)
            batch_x.append(make_sample(item.state, item.action))

        return (np.array(batch_x), np.array(batch_y))


class DQN:
    def __init__(self):
        input = tf.keras.layers.Input(shape=(5,))
        layer = tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)(input)
        layer = tf.keras.layers.Dense(1, activation=None)(layer)

        learning_rate = 0.001
        model = tf.keras.Model(inputs=[input], outputs=[layer])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.MeanSquaredError()
        )
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict_q(self, state) -> np.ndarray:
        batch = np.array([
            make_sample(state, 0),
            make_sample(state, 1),
        ])
        return self(batch)

    def predict_action(self, state) -> int:
        return int(np.argmax(self.predict_q(state)))

    def train(self, batch_x: np.ndarray, batch_y: np.ndarray):
        self.model.train_on_batch(batch_x, batch_y)


predict_net = DQN()
train_net = DQN()

RENDER = True

epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.002  # Exponential decay rate for exploration prob
discount_rate = 0.95

episode_count = 2000
max_episode_steps = 500
episode_report_n = 250
episode_weight_sync_n = 20
rewards_accum = []
buffer = ReplayBuffer()

with gym.make("CartPole-v1", render_mode="rgb_array") as env:
    env.action_space.seed(42)
    state, info = env.reset(seed=42)

    for episode in tqdm(range(episode_count)):
        episode_is_report = episode % episode_report_n == 0
        if episode % episode_weight_sync_n == 0:
            predict_net.model.set_weights(train_net.model.get_weights())

        state, _ = env.reset()
        episode_steps = 0
        for step in range(max_episode_steps):
            exp_exp_tradeoff = random.uniform(0, 1)

            # If this number > greater than epsilon --> exploitation (taking the biggest Q value
            # for this state)
            if exp_exp_tradeoff > epsilon:
                action = predict_net.predict_action(state)
            else:
                # Else doing a random choice --> exploration
                action = int(env.action_space.sample())

            new_state, reward, terminated, truncated, _ = env.step(action)
            stop = terminated or truncated
            buffer.push(StepHistoryItem(
                action=action,
                state=state,
                next_state=new_state if not stop else None
            ))
            state = new_state

            if episode_is_report and RENDER:
                img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow("CartPole", img)
                cv2.waitKey(50)

            episode_steps += 1
            if stop:
                break

        batch_x, batch_y = buffer.generate_batch(64, predict_net, discount_rate)
        train_net.train(batch_x, batch_y)

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        rewards_accum.append(episode_steps)

        if episode_is_report or episode == episode_count - 1:
            print(
                f"Episode {episode}: rewards sum: {sum(rewards_accum)}, mean: {np.mean(rewards_accum)}, min: {np.min(rewards_accum)}, max: {np.max(rewards_accum)}, epsilon: {epsilon}")
            rewards_accum.clear()

tf.keras.models.save_model(predict_net.model, "dqn.hdf5")
