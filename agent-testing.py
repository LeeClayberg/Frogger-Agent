import os
from collections import deque
import random
from ale_py import ALEInterface
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

save_files = sorted(list(filter(lambda x: "trial_weights" in x, os.listdir('.'))))

exploration_rate = 0.25
exploration_min = 0.01
exploration_decay = 0.995
gamma = 0.95
learning_rate = 0.001

batch_size = 32
episodes = 20

memory = deque(maxlen=2000)

ale = ALEInterface()

# Load the ROM file
ale.setInt("random_seed", 123)
ale.loadROM("frogger.bin")

# Get the list of legal actions
legal_actions = ale.getMinimalActionSet()

lives_count = ale.lives()
forward_count = 0
is_past_road = True
extra_action_rewards = {0: -0.5, 1: 0.0, 2: 0.0, 3: 0.0, 4: -1.0}

sample_screen = ale.getScreen().reshape([210, 160])
sample_screen = sample_screen[::4, ::4]
input_dim = np.size(sample_screen)

# Model
model = tf.keras.Sequential()
model.add(layers.Dense(128, input_dim=input_dim))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(legal_actions), activation='linear'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

for weight_file in save_files:
    model.load_weights(weight_file)
    avg_reward = 0
    # Play episodes
    for episode in range(episodes):
        actual_reward = 0
        while not ale.game_over():
            # Action
            state = ale.getScreen().reshape([210, 160])
            state = state[::4, ::4]
            state = state.reshape([1, input_dim])
            if random.random() <= exploration_rate:
                action = random.randrange(len(legal_actions))
            else:
                values = model.predict(state)
                action = np.argmax(values[0])
            reward = ale.act(legal_actions[action])
            actual_reward += reward
        print(f"  Episode {episode} ended with score: {actual_reward}")
        avg_reward += (actual_reward / episodes)
        ale.reset_game()
    print(f"{weight_file} average: {avg_reward}")
