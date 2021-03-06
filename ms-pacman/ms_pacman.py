## Léo Poulin
## 017465433

import os
import time
import random
import argparse
import datetime
import numpy as np

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam, RMSprop
from progressbar import ProgressBar, AdaptiveETA, Bar, SimpleProgress

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import gym

FRAME_CUT = 171

class GameEnvironment():
    def __init__(self, env_name: str):
        self.env = gym.make(env_name)

class MsPacman(GameEnvironment):
    def __init__(self, model: str, learning_rate: float, exploration, render: bool, iterations: int):
        super().__init__('MsPacman-v0')
        self.weights_file_basename = 'w_mspacman'
        self.render = render
        self.iterations = iterations
        self.memory = deque(maxlen=50000)
        self.observation_loops = 10
        self.batch_size = 32
        self.learning_rate = learning_rate
        self.discount_rate = 0.9
        self.rms_rho = 0.95
        self.exploration_rate = exploration[0]
        self.explotation_rate_min = exploration[1]
        self.exploration_decay = exploration[2]
        self.exploration_hist = list()
        self.observe_fitness_score = list()
        self.play_fitness_score = list()
        self.fit_loss = list()
        ## Creating the model
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
        # model.add(Conv2D(32, (8, 8), strides=4,
        #     input_shape=self.env.observation_space.shape,
        #     activation='relu'))
        model.add(Conv2D(1, (1, 1), strides=1,
            input_shape=self.env.observation_space.shape,
            activation='relu'))
        model.add(Conv2D(32, (8, 8), strides=4, activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=1, activation='relu'))
        model.add(Flatten())    # Important to have a 1D output
        model.add(Dense(512, activation='relu'))
        model.add(Dense(5, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=RMSprop(
                lr=self.learning_rate,
                rho=self.rms_rho,
                decay=0))
        return model

    def load(self, name: str=None):
        if os.path.exists(name):
            self.model.load_weights(name)
            print("Loaded weights file successfully")
        else:
            print("Weight file '{}' not found".format(name))

    def save(self, end_time: datetime.datetime):
        self.model.save_weights("{}_{}it_{}.h5".format(
            self.weights_file_basename,
            self.iterations,
            end_time.strftime("%m%d%H%M%S")))

    def summary(self) -> None:
        self.model.summary()

    def observe(self) -> None:
        nb_games_done = 0
        game_reward = 0
        obs = self.env.reset()
        state = np.expand_dims(obs, axis=0)
        best_fitness_score = 0
        average_fitness_score = 0
        self.exploration_hist.append(self.exploration_rate)
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
                average_fitness_score += game_reward
                if game_reward > best_fitness_score:
                    best_fitness_score = game_reward
                game_reward = 0
                obs = self.env.reset()                                      # Restart the game
                state = np.expand_dims(obs, axis=0)
        average_fitness_score /= self.observation_loops
        self.observe_fitness_score.append(average_fitness_score)
        # self.observe_fitness_score.append(best_fitness_score)

    def learn_from_replay(self) -> None:
        loss = list()
        replay_batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in replay_batch:
            target = reward
            if not done:
                target += reward + self.discount_rate * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            loss.append(self.model.fit(state, target_f, epochs=1, verbose=0).history['loss'][0])
        self.fit_loss.append(np.mean(loss))
        if self.exploration_rate > self.explotation_rate_min:
            self.exploration_rate *= self.exploration_decay

    def forget(self) -> None:
        self.memory.clear()

    def play(self, slow: bool=False) -> None:
        obs = self.env.reset()
        state = np.expand_dims(obs, axis=0)
        done = False
        tt_reward = 0
        while not done:
            if self.render:
                self.env.render()
            q_values = self.model.predict(state)
            action = np.argmax(q_values)
            obs_new, reward, done, _ = self.env.step(action)
            state = np.expand_dims(obs_new, axis=0)
            tt_reward += reward
            if slow:
                time.sleep(0.033)
        self.play_fitness_score.append(tt_reward)

    def draw_fitness_stats(self, end_time: datetime.datetime):
        plt.clf()
        dql_patch = mpatches.Patch(color='red', label='DQL Agent')
        r_patch = mpatches.Patch(color='black', label='Average Random Agent')
        plt.legend(handles=[dql_patch, r_patch])
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Score')
        plt.plot(self.observe_fitness_score, 'k')
        plt.plot(self.play_fitness_score, 'r')
        plt.savefig("fitness_{}it_{}".format(self.iterations, end_time.strftime("%m%d%H%M%S")))

    def draw_exploration_decay(self, end_time: datetime.datetime):
        plt.clf()
        plt.ylim(0, 1)
        plt.plot(self.exploration_hist, 'g')
        plt.ylabel('Exploration rate')
        plt.xlabel('Iteration')
        plt.savefig("exploration_{}it_{}".format(self.iterations, end_time.strftime("%m%d%H%M%S")))

    def draw_loss(self, end_time: datetime.datetime):
        plt.clf()
        plt.yscale('log')
        plt.plot(self.fit_loss, 'b')
        plt.ylabel('Loss (Mean Absolute Error)')
        plt.xlabel('Iteration')
        plt.savefig("loss_{}it_{}".format(self.iterations, end_time.strftime("%m%d%H%M%S")))
        

def main():
    parser = argparse.ArgumentParser(
        prog='ms_pacman.py',
        description='DQN Agent that plays mspacman')
    parser.add_argument(
        '-r', '--render',
        action='store_true',
        help='render the graphical environment')
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
        help='Load a weight file')
    parser.add_argument(
        '-s', '--save',
        action='store_true',
        help="save the trained weights into a file 'basename+timestamp'.h5")
    parser.add_argument(
        '-L', '--learning-rate',
        type=float,
        action='store',
        default=0.00025,
        help='Learning rate of the optimizer')
    parser.add_argument(
        '-e', '--exploration',
        nargs=3,
        type=float,
        metavar=['START', 'MIN', 'DECAY'],
        action='store',
        default=(1.0, 0.1, 0.995),
        help="exploration rate starting value, minimum value, decay")
    parser.add_argument(
        '-p', '--play',
        action='store_true',
        help='play only (no learning)')
    args = parser.parse_args()
    bar = ProgressBar(
        max_value=args.iterations,
        widgets=['Iteration ', SimpleProgress(), ' ', Bar(), ' ', AdaptiveETA()],
        redirect_stdout=True)
    agent = MsPacman(
        args.model,
        args.learning_rate,
        args.exploration,
        args.render,
        args.iterations)
    agent.summary()
    if args.load:
        agent.load(args.load)
    if args.play:
        agent.play(True)
        print("Score: {}".format(agent.play_fitness_score[0]))
        return
    bar.start()
    for _ in range(agent.iterations):
        agent.observe()
        agent.learn_from_replay()
        agent.play()
        bar += 1
    bar.finish()
    end_time = datetime.datetime.now()
    agent.draw_fitness_stats(end_time)
    agent.draw_exploration_decay(end_time)
    agent.draw_loss(end_time)
    if args.save:
        agent.save(end_time)
        
if __name__ == "__main__":
    main()
