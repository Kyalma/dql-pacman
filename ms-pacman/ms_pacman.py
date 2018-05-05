## Léo Poulin

import time

from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from keras.optimizers import Adam
from collections import deque            # For storing moves 

import random     # For sampling batches from the observations
import numpy as np
import gym                                # To train our network

GAME_OBSERVETIME = 5
BATCH_SIZE = 32

OBSERVETIME = 2000                          # Number of timesteps we will be acting on the game and observing results
RANDOM_THRESHOLD = 0.6                              # Probability of doing a random move
GAMMA = 0.9                                # Discounted future reward. How much we care about steps further in time
MINIBATCH_SIZE = 50                               # Learning minibatch size

class MsPacman():
    def __init__(self):
        self.env = gym.make('MsPacman-v0')
        self.memory = deque()
        self.learning_rate = 0.01
        self.exploration_rate = 0.7
        self.explotation_rate_min = 0.1
        self.exploration_decay = 0.9
        ## Creating the model
        self.model = Sequential()
        self.model.add(Conv2D(16, (3, 3),                       # Conv 2D
            padding='same',
            input_shape=self.env.observation_space.shape))
        self.model.add(Activation('relu'))                      #  * relu
        self.model.add(Conv2D(32, (3, 3)))                      # Conv 2D
        self.model.add(Activation('relu'))                      #  * relu
        self.model.add(MaxPooling2D(pool_size=(2, 2)))          # MaxPooling(2, 2)
        self.model.add(Dropout(0.6))                            #  * Dropout
        self.model.add(Conv2D(32, (3, 3), padding='same'))      # Conv2D
        self.model.add(Activation('relu'))                      #  * relu
        self.model.add(Conv2D(32, (3, 3)))                      # Conv2D
        self.model.add(Activation('relu'))                      #  * relu
        self.model.add(MaxPooling2D(pool_size=(2, 2)))          # MaxPooling(2, 2)
        self.model.add(Dropout(0.2))                            #  * Dropout
        self.model.add(Flatten())                               # Flatten
        self.model.add(Dense(512))                              # Dense
        self.model.add(Activation('softmax'))                      #  * relu
        self.model.add(Dense(5,
            activation='softmax', init='uniform'))              # Dense (Output)
        # self.model.add(Activation('softmax'))                 #  * softmax
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def summary(self) -> None:
        self.model.summary()

    def observe(self) -> None:
        done = True
        nb_games_done = 0
        game_reward = 0
        obs = self.env.reset()
        state = np.expand_dims(obs, axis=0)
        print("Observing Game 1")
        while nb_games_done < GAME_OBSERVETIME:
            self.env.render()
            if np.random.rand() <= self.exploration_rate:
                action = np.random.randint(0, 4, size=1)[0]
            else:
                q_values = self.model.predict(state)                     # Q-values predictions
                action = np.argmax(q_values)                        # Move with highest Q-value is the chosen one
            obs_new, reward, done, _ = self.env.step(action)          # See state of the game, reward... after performing the action
            game_reward += reward
            state_new = np.expand_dims(obs_new, axis=0)
            self.memory.append((state, action, reward, state_new, done))          # 'Remember' action and consequence
            # state = np.append(state_new, state, axis=0)
            state = state_new                                       # Update state
            if done:
                print("Total reward for this game: {}".format(game_reward))
                nb_games_done += 1
                game_reward = 0
                obs = self.env.reset()                                   # Game begins
                state = np.expand_dims(obs, axis=0)                 # (Formatting issues) Making the observation the first element of a batch of inputs 
                print("Observing Game {}".format(nb_games_done + 1))
        print('Observation done')

    def learn_from_replay(self) -> None:
        replay_batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in replay_batch:
            if not done:
                optimal_future_value = np.amax(self.model.predict(next_state)[0])
                target = reward + GAMMA * optimal_future_value
            else:
                target = reward
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1)
        if self.exploration_rate > self.explotation_rate_min:
            self.exploration_rate *= self.exploration_decay

    def play(self) -> None:
        obs = self.env.reset()
        state = np.expand_dims(obs, axis=0)
        done = False
        tot_reward = 0.0
        while not done:
            self.env.render()
            Q = self.model.predict(state)
            action = np.argmax(Q)       
            obs_new, reward, done, _ = self.env.step(action)
            state = np.expand_dims(obs_new, axis=0)
            tot_reward += reward
        print('Game ended! Total reward: {}'.format(tot_reward))


# def learn(model, register, state):
#     # SECOND STEP: Learning from the observations (Experience replay)
#     minibatch = random.sample(register, MINIBATCH_SIZE)         # Sample some moves
#     inputs_shape = (MINIBATCH_SIZE,) + state.shape[1:]
#     inputs = np.zeros(inputs_shape)
#     targets = np.zeros((MINIBATCH_SIZE, ENV.action_space.n))
#     for i in range(0, MINIBATCH_SIZE):
#         state = minibatch[i][0]
#         action = minibatch[i][1]
#         reward = minibatch[i][2]
#         state_new = minibatch[i][3]
#         done = minibatch[i][4]        
#     # Build Bellman equation for the Q function
#         inputs[i:i+1] = np.expand_dims(state, axis=0)
#         targets[i] = model.predict(state)
#         Q_sa = model.predict(state_new)
        
#         if done:
#             targets[i, action] = reward
#         else:
#             targets[i, action] = reward + GAMMA * np.max(Q_sa)
#     # Train network to output the Q function
#         model.train_on_batch(inputs, targets)
#     print('Learning Finished')

def main():
    agent = MsPacman()
    agent.summary()
    # agent.model.load_weights('mspacman-weights.h5')
    for iteration in range(10):
        print("Iteration {}".format(iteration))
        agent.observe()
        agent.learn_from_replay()
        agent.play()
        agent.memory.clear()
        agent.model.save_weights('mspacman-weights.h5')

if __name__ == "__main__":
    # test_play()
    main()
