import os
from collections import deque
import random
from ale_py import ALEInterface
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Load file
iteration = 6
load_file = f"saved_weights/trial_weights_{iteration}.h5"

exploration_rate = 0.1
learning_rate = 0.001

episodes = 10

ale = ALEInterface()

# Load the ROM file
ale.setInt("random_seed", 123)
ale.loadROM("frogger.bin")

# Get the list of legal actions
legal_actions = ale.getMinimalActionSet()

# Extra parameters
lives_count = ale.lives()
is_past_road = True
extra_action_rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

# Screen size based on model
if iteration < 5:
    sample_screen = ale.getScreen().reshape([210, 160])
    sample_screen = sample_screen[::4, ::4]
    input_dim = np.size(sample_screen)
else:
    sample_screen = ale.getScreen().reshape([210, 160])
    sample_screen = sample_screen[92:188, 8:151]
    sample_screen = sample_screen[::2, ::2]
    input_dim = np.size(sample_screen)

# Model
model = tf.keras.Sequential()
model.add(layers.Dense(128, input_dim=input_dim))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(legal_actions), activation='linear'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

# Load file weights
if os.path.isfile(load_file):
    model.load_weights(load_file)

# Play episodes
os.makedirs('images')
for episode in range(episodes):
    if not os.path.exists(f'images/episode-{episode:02d}'):
        os.makedirs(f'images/episode-{episode:02d}')
    total_reward = 0
    actual_reward = 0
    lives_count = ale.lives()
    is_past_road = True
    has_started = False
    frame = 0
    # Main game loop
    while not ale.game_over():
        # Action
        if iteration < 5:
            state = ale.getScreen().reshape([210, 160])
            state = state[::4, ::4]
        else:
            state = ale.getScreen().reshape([210, 160])
            state = state[92:188, 8:151]
            state = state[::2, ::2]
        state = state.reshape([1, input_dim])
        if random.random() <= exploration_rate:
            action = random.randrange(len(legal_actions))
        else:
            values = model.predict(state)
            action = np.argmax(values[0])
        # Get reward
        reward = ale.act(legal_actions[action])
        if reward > 0:
            has_started = True
        if has_started:
            # Additional rewards
            actual_reward += reward
            reward += extra_action_rewards[action]
            if 214 in ale.getScreen().reshape([210, 160])[97, 8:151] and not is_past_road:
                reward = 150
                is_past_road = True
            # Next Life
            if lives_count > ale.lives():
                if not is_past_road:
                    reward = -100
                is_past_road = False
                lives_count = ale.lives()
            total_reward += reward
            frame += 1
            # Save frame as image
            ale.saveScreenPNG(f"images/episode-{episode:02d}/frame-{frame:04d}.jpg")
    print(f"Episode {episode} ended with score: {actual_reward} New Reward: {total_reward}")
    # Reset game
    ale.reset_game()
