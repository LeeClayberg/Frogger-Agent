import os
import random
from ale_py import ALEInterface
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

save_files = sorted(os.listdir('./saved_weights'))

exploration_rate = 0.1
learning_rate = 0.001

episodes = 50

ale = ALEInterface()

# Load the ROM file
ale.setInt("random_seed", 123)
ale.loadROM("frogger.bin")

# Get the list of legal actions
legal_actions = ale.getMinimalActionSet()

# Extra parameters
lives_count = ale.lives()
forward_count = 0
is_past_road = True
extra_action_rewards = {0: -0.5, 1: 0.0, 2: 0.0, 3: 0.0, 4: -1.0}

# Get size of screen
sample_screen = ale.getScreen().reshape([210, 160])
sample_screen = sample_screen[::4, ::4]
input_dim = np.size(sample_screen)

# Model
model = tf.keras.Sequential()
model.add(layers.Dense(128, input_dim=input_dim))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(legal_actions), activation='linear'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

# First four models
for weight_file in save_files[:4]:
    model.load_weights(f"saved_weights/{weight_file}")
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
        # Reset game
        ale.reset_game()
    # Average reward across all games
    print(f"{weight_file} average: {avg_reward}")

# Get new size of screen
sample_screen = ale.getScreen().reshape([210, 160])
sample_screen = sample_screen[92:188, 8:152]
sample_screen = sample_screen[::2, ::2]
input_dim = np.size(sample_screen)

# Second model
model2 = tf.keras.Sequential()
model2.add(layers.Dense(128, input_dim=input_dim))
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(len(legal_actions), activation='linear'))

model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

# Models five and above
for weight_file in save_files[4:]:
    model2.load_weights(f"saved_weights/{weight_file}")
    avg_reward = 0
    # Play episodes
    for episode in range(episodes):
        actual_reward = 0
        while not ale.game_over():
            # Action
            state = ale.getScreen().reshape([210, 160])
            state = state[92:188, 8:152]
            state = state[::2, ::2]
            state = state.reshape([1, input_dim])
            if random.random() <= exploration_rate:
                action = random.randrange(len(legal_actions))
            else:
                values = model2.predict(state)
                action = np.argmax(values[0])
            reward = ale.act(legal_actions[action])
            actual_reward += reward
        print(f"  Episode {episode} ended with score: {actual_reward}")
        avg_reward += (actual_reward / episodes)
        # Reset game
        ale.reset_game()
    # Average reward across all games
    print(f"{weight_file} average: {avg_reward}")
