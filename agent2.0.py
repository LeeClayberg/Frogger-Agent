import os
from collections import deque
import random
from ale_py import ALEInterface
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# For weight file 5
SAVE_FILE = "saved_weights/trial_weights_5.h5"

exploration_rate = 1.0
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

# extra_action_rewards = {0: -0.5, 1: 0.0, 2: 0.0, 3: 0.0, 4: -1.0}
extra_action_rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

# Get size of screen
sample_screen = ale.getScreen().reshape([210, 160])
sample_screen = sample_screen[92:188, 8:151]
sample_screen = sample_screen[::2, ::2]
input_dim = np.size(sample_screen)
print(input_dim)

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
    score = 0
    lives_count = ale.lives()
    is_past_road = False
    has_started = False
    big_rewards = []
    # Main game loop
    while not ale.game_over():
        # Action
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
            score += reward
            reward += extra_action_rewards[action]
            if 214 in ale.getScreen().reshape([210, 160])[97, 8:151] and not is_past_road:
                reward = 75
                big_rewards.append(reward)
                is_past_road = True
            # Next Life
            if lives_count > ale.lives():
                if not is_past_road:
                    reward = -50
                    big_rewards.append(reward)
                is_past_road = False
                lives_count = ale.lives()
            # Add to memory
            memory.append((state, action, reward, ale.game_over()))
            total_reward += reward
    print(f"Episode {episode}\t ended with score: {score}\t New Reward: {total_reward}\tExploration: {exploration_rate}\tL: [{', '.join(map(lambda x: str(x), big_rewards))}]")
    # Update model
    if len(memory) >= batch_size:
        # Sample memory
        batch = random.sample(memory, batch_size)
        # Adjust model
        for state, action, reward, done in batch:
            target_f = model.predict(state)
            target_f[0][action] = reward
            model.fit(state, target_f, epochs=1, verbose=0)
    # Reset game
    ale.reset_game()

# Save weights
model.save(SAVE_FILE)
