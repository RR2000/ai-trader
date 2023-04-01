import time

import tensorflow as tf
import numpy as np
from collections import deque
import random


class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)  # 2000
        self.gamma = 0.95
        self.epsilon = 1  # 1.0
        self.epsilon_min = 0.10  # 0.01
        self.epsilon_decay = 0.99995  # 0.995
        self.learning_rate = 0.001  # 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(51, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(11, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            with tf.GradientTape() as tape:
                predictions = self.model(state, training=True)
                loss = tf.keras.losses.MSE(target_f, predictions)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


def train_dqn(env, episodes, batch_size, max_hours=999999):
    max_seconds = max_hours * 3600  # Convert max_hours to seconds
    start_time = time.time()

    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    dqn = DQN(state_size, action_size)

    state = env.reset()
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        count = 0

        while not done:
            count += 1
            action = dqn.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            dqn.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(
                    f"Episode: {e + 1}/{episodes}, Steps: {count}, Total Reward: {total_reward}, Epsilon:{dqn.epsilon}",
                    flush=True)

            if len(dqn.memory) > batch_size:
                dqn.replay(batch_size)

        env.render()

        if max_hours > 0 and time.time() - start_time > max_seconds:
            print(f"Stopping training due to time limit of {max_hours} hours.")
            break

        if e % 20 == 0:
            dqn.model.save(f'src/model/checkpoint/trading_model_episode_{e}.h5')

    return dqn.model

