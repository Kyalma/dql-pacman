## Léo Poulin
## 017465433

import os
import time

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam
from collections import deque

import random
import numpy as np
import gym


class MsPacman():
    def __init__(self):
        self.env = gym.make('MsPacman-v0')
        self.weights_file = './ms-pacman-w.h5'
        self.memory = deque()
        self.observation_loops = 5
        self.batch_size = 32
        self.learning_rate = 0.001
        self.exploration_rate = 0.7
        self.explotation_rate_min = 0.1
        self.exploration_decay = 0.9
        self.discount_rate = 0.9
        ## Creating the model
        self.model = Sequential()
        self.__build_model()

    def __build_model(self):
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
        self.model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate),
            metrics=['mae'])

    def load(self):
        if os.path.exists(self.weights_file):
            self.model.load_weights(self.weights_file)
            print("Loaded weights file successfully")
        else:
            print("Weight file '{}' not found".format(self.weights_file))

    def save(self):
        self.model.save_weights(self.weights_file)

    def summary(self) -> None:
        self.model.summary()

    def observe(self) -> None:
        done = True
        nb_games_done = 0
        game_reward = 0
        obs = self.env.reset()
        state = np.expand_dims(obs, axis=0)
        print("Observing Game 1")
        while nb_games_done < self.observation_loops:
            self.env.render()
            if np.random.rand() <= self.exploration_rate:
                action = np.random.randint(0, 4, size=1)[0]
            else:
                q_values = self.model.predict(state)                        # Q-values predictions
                action = np.argmax(q_values)                                # Choose the move with the highest reward from the prediction
            obs_new, reward, done, _ = self.env.step(action)
            game_reward += reward
            state_new = np.expand_dims(obs_new, axis=0)
            self.memory.append((state, action, reward, state_new, done))    # 'Remember' action and consequence
            state = state_new                                               # Update state
            if done:
                print("Total reward for this game: {}".format(game_reward))
                nb_games_done += 1
                game_reward = 0
                obs = self.env.reset()                                      # Restart the game
                state = np.expand_dims(obs, axis=0)
                print("Observing Game {}".format(nb_games_done + 1))
        print('Observation done')

    def learn_from_replay(self) -> None:
        replay_batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in replay_batch:
            if not done:
                optimal_future_value = np.amax(self.model.predict(next_state)[0])
                target = reward + self.discount_rate * optimal_future_value
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
            q_values = self.model.predict(state)
            action = np.argmax(q_values)
            obs_new, reward, done, _ = self.env.step(action)
            state = np.expand_dims(obs_new, axis=0)
            tot_reward += reward
        print('Game ended! Total reward: {}'.format(tot_reward))


def main():
    agent = MsPacman()
    agent.summary()
    agent.load()
    for iteration in range(10):
        print("Iteration {}".format(iteration))
        agent.observe()
        agent.learn_from_replay()
        agent.play()
        agent.save()
        agent.memory.clear()
        
if __name__ == "__main__":
    main()
