import os
from collections import deque
import random
from ale_py import ALEInterface
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

SAVE_FILE = "saved_weights/trial_weights_4.h5"

exploration_rate = 0.0487
exploration_min = 0.01
exploration_decay = 0.995
gamma = 0.95
learning_rate = 0.001

batch_size = 32
episodes = 1

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

if os.path.isfile(SAVE_FILE):
    model.load_weights(SAVE_FILE)
    # exploration_rate = exploration_min

# Play episodes
for episode in range(episodes):
    if not os.path.exists(f'images/episode-{episode:02d}'):
        os.makedirs(f'images/episode-{episode:02d}')
    total_reward = 0
    actual_reward = 0
    has_started = False
    frame = 0
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
        reward = ale.act(legal_actions[action]) # Gamma of 0
        if reward > 0:
            has_started = True
        if has_started:
            if action == 1:
                forward_count += 1
            if action == 4:
                forward_count -= 1
            actual_reward += reward
            reward += extra_action_rewards[action]
            if forward_count == 6 and not is_past_road:
                reward = 100
                is_past_road = True
            # Next Life
            if lives_count > ale.lives():
                is_past_road = False
                lives_count = ale.lives()
                forward_count = 0
                reward = -100
            memory.append((state, action, reward, ale.game_over()))
            total_reward += reward
            frame += 1
            ale.saveScreenPNG(f"images/episode-{episode:02d}/frame-{frame:04d}.jpg")
    print(f"Episode {episode} ended with score: {actual_reward} New Reward: {total_reward}")
    ale.reset_game()
