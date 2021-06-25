import os
from collections import deque
import random
from ale_py import ALEInterface
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# For weight files 1-4
SAVE_FILE = "saved_weights/trial_weights_4.h5"

exploration_rate = 1.0
exploration_min = 0.5
exploration_decay = 0.995
learning_rate = 0.001

batch_size = 32
episodes = 1000

memory = deque(maxlen=2000)

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

# Load weights if the weights have been previously calculated
if os.path.isfile(SAVE_FILE):
    model.load_weights(SAVE_FILE)

# Play episodes
for episode in range(episodes):
    total_reward = 0
    actual_reward = 0
    counter = 0
    has_started = False
    rewards = []
    # Main game loop
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
        # Get reward
        reward = ale.act(legal_actions[action])
        if reward > 0:
            has_started = True
        if has_started:
            # Additional rewards
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
            rewards.append(reward)
            # Add to memory
            memory.append((state, action, reward, ale.game_over()))
            total_reward += reward
    print(f"Episode {episode} ended with score: {actual_reward} New Reward: {total_reward}\tExploration: {exploration_rate}\tL: [{' '.join(map(lambda x: str(x), rewards))}]")
    # Update model
    if len(memory) >= batch_size:
        # Sample memory
        batch = random.sample(memory, batch_size)
        # Adjust model
        for state, action, reward, done in batch:
            target_f = model.predict(state)
            target_f[0][action] = reward
            model.fit(state, target_f, epochs=1, verbose=0)
        if exploration_rate > exploration_min:
            exploration_rate *= exploration_decay
    # Reset game
    ale.reset_game()

# Save weights
model.save(SAVE_FILE)
