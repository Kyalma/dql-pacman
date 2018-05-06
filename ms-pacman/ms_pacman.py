## Léo Poulin
## 017465433

import os
import time
import random
import argparse
import datetime
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout, LeakyReLU
from keras.optimizers import Adam, RMSprop
from collections import deque

import tkinter
import matplotlib.pyplot as plt

import gym


class MsPacman():
    def __init__(self, model: str, weights_file_basename: str, render: bool, iterations: int):
        self.env = gym.make('MsPacman-v0')
        self.weights_file_basename = weights_file_basename
        self.render = render
        self.iterations = iterations
        self.memory = deque()
        self.observation_loops = 10
        self.batch_size = 32
        self.learning_rate = 0.00025
        self.exploration_rate = 1
        self.explotation_rate_min = 0.05
        self.exploration_decay = 0.9
        self.discount_rate = 0.9
        self.observe_fitness_score = list()
        self.play_fitness_score = list()
        self.rms_rho = 0.95
        ## Creating the model
        # self.model = self._build_alexnet_based_model()
        if model == 'deepmind':
            self.model = self._build_deepmind_model()
        elif model == 'alexnet':
            self.model = self._build_alexnet_based_model()
        else:
            raise NameError('Model not defined')

    def _build_alexnet_based_model(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3),                       # Conv 2D
            padding='same',
            input_shape=self.env.observation_space.shape))
        model.add(Activation('relu'))                      #  * relu
        model.add(Conv2D(32, (3, 3)))                      # Conv 2D
        model.add(Activation('relu'))                      #  * relu
        model.add(MaxPooling2D(pool_size=(2, 2)))          # MaxPooling(2, 2)
        model.add(Dropout(0.6))                            #  * Dropout
        model.add(Conv2D(32, (3, 3), padding='same'))      # Conv2D
        model.add(Activation('relu'))                      #  * relu
        model.add(Conv2D(32, (3, 3)))                      # Conv2D
        model.add(Activation('relu'))                      #  * relu
        model.add(MaxPooling2D(pool_size=(2, 2)))          # MaxPooling(2, 2)
        model.add(Dropout(0.2))                            #  * Dropout
        model.add(Flatten())                               # Flatten
        model.add(Dense(512))                              # Dense
        model.add(Activation('softmax'))                      #  * relu
        model.add(Dense(5,
            activation='softmax', init='uniform'))              # Dense (Output)
        # self.model.add(Activation('softmax'))                 #  * softmax
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate),
            metrics=['mae'])
        return model

    def _build_deepmind_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8),
            strides=4,
            input_shape=self.env.observation_space.shape,
            activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(5, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=RMSprop(lr=self.learning_rate, rho=self.rms_rho, decay=1e-06))
        return model

    def load(self):
        if os.path.exists(self.weights_file_basename):
            self.model.load_weights(self.weights_file_basename)
            print("Loaded weights file successfully")
        else:
            print("Weight file '{}' not found".format(self.weights_file_basename))

    def save(self, curr_time):
        self.model.save_weights("{}_{}.h5".format(self.weights_file_basename, curr_time))

    def summary(self) -> None:
        self.model.summary()

    def observe(self) -> None:
        done = True
        nb_games_done = 0
        game_reward = 0
        obs = self.env.reset()
        state = np.expand_dims(obs, axis=0)
        best_fitness_score = 0
        while nb_games_done < self.observation_loops:
            if self.render:
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
                nb_games_done += 1
                if game_reward > best_fitness_score:
                    best_fitness_score = game_reward
                game_reward = 0
                obs = self.env.reset()                                      # Restart the game
                state = np.expand_dims(obs, axis=0)
        self.observe_fitness_score.append(best_fitness_score)

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
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.explotation_rate_min:
            self.exploration_rate *= self.exploration_decay

    def forget(self) -> None:
        self.memory.clear()

    def play(self) -> None:
        obs = self.env.reset()
        state = np.expand_dims(obs, axis=0)
        done = False
        tt_reward = 0.0
        while not done:
            if self.render:
                self.env.render()
            q_values = self.model.predict(state)
            action = np.argmax(q_values)
            obs_new, reward, done, _ = self.env.step(action)
            state = np.expand_dims(obs_new, axis=0)
            tt_reward += reward
        self.play_fitness_score.append(tt_reward)

    def draw_fitness_stats(self, curr_time):
        plt.xticks(range(self.iterations))
        plt.xlabel('Nb of Iterations')
        plt.ylabel('Fitness Score')
        plt.plot(self.observe_fitness_score, 'k')
        plt.plot(self.play_fitness_score, 'r')
        plt.savefig("fitness_{}it_{}".format(self.iterations, curr_time))


def main():
    parser = argparse.ArgumentParser(
        prog='ms_pacman.py',
        description='DQN Agent that plays mspacman')
    parser.add_argument(
        '-r', '--render',
        action='store_true',
        help='render the graphical environment')
    parser.add_argument(
        '-b', '--basename',
        action='store',
        type=str,
        help='give a path to the .h5 file to load/save',
        default='ms-pacman-w')
    parser.add_argument(
        '-m', '--model',
        action='store',
        type=str,
        default='deepmind',
        help='Choose a model between alexnet and deepmind')
    parser.add_argument(
        '-i', '--iterations',
        type=int,
        action='store',
        default=10,
        help='How much time the program will loop')
    parser.add_argument(
        '-l', '--load',
        type=str,
        action='store',
        default=None,
        help='Load a weight file'
    )
    parser.add_argument(
        "-s", "--save",
        action='store_true',
        help="save the trained weights into a file 'basename+timestamp'.h5")

    args = parser.parse_args()
    agent = MsPacman(args.model, args.weights, args.render, args.iterations)
    agent.summary()
    if args.load:
        agent.load(args.load)
    for iteration in range(agent.iterations):
        print("Iteration {}/{}".format(iteration, agent.iterations))
        agent.observe()
        agent.learn_from_replay()
        agent.play()
        agent.forget()
    time = datetime.datetime.now().strftime("%m%d%H%M%S")
    print("Best observed scores : {}".format(agent.observe_fitness_score))
    print("Play scores          : {}".format(agent.play_fitness_score))
    agent.draw_fitness_stats(time)
    if args.save:
        agent.save(time)
        
if __name__ == "__main__":
    main()
