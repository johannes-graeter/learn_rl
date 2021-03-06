import tensorflow as tf
from collections import deque
import numpy as np
import random
from itertools import islice


# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
        self.learning_rate = 0.001
        self.acted_randomly = True
        self.model = self._build_model()

    def _build_model(self):
        # Convolutional Neural Net for Deep-Q learning Model from images.
        # TODO: Maybe use pretrained model and retrain (f.e. MobileNet) https://keras.io/applications/
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=32, input_shape=self.state_size, kernel_size=(7, 7), strides=(3, 3),
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, states, action, reward, next_state, done):
        self.memory.append((states, action, reward, next_state, done))

    def act(self, states):
        self.acted_randomly = False
        if np.random.rand() <= self.epsilon:
            self.acted_randomly = True
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([np.concatenate(states, axis=2)]))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for states, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                next_input = np.array(
                    [np.concatenate((next_state, *deque(islice(states, 0, len(states) - 1))), axis=2)])
                target = reward + self.gamma * np.amax(self.model.predict(next_input))
                cur_input = np.array([np.concatenate(states, axis=2)])
                target_f = self.model.predict(cur_input)
                target_f[0][action] = target
                self.model.fit(cur_input, target_f, epochs=1, verbose=0)
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

    def save(self, path):
        self.model.save(path)
