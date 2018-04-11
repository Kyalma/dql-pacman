## Léo Poulin

from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            # For storing moves 

import numpy as np
import gym                                # To train our network
env = gym.make('MsPacman-v0')          # Choose game (any in the gym should work)

import random     # For sampling batches from the observations

OBSERVETIME = 1200                          # Number of timesteps we will be acting on the game and observing results
EPSILON = 0.8                              # Probability of doing a random move
GAMMA = 0.9                                # Discounted future reward. How much we care about steps further in time
MINIBATCH_SIZE = 50                               # Learning minibatch size

def create_model():
    # Create network. Input is two consecutive game states, output is Q-values of the possible moves.
    env_shape = (2, ) + env.observation_space.shape
    model = Sequential()
    model.add(Dense(20, input_shape=env_shape, init='uniform', activation='relu'))
    model.add(Flatten())       # Flatten input so as to have no problems with processing
    model.add(Dense(18, init='uniform', activation='relu'))
    model.add(Dense(10, init='uniform', activation='relu'))
    model.add(Dense(env.action_space.n, init='uniform', activation='linear'))    # Same number of outputs as possible actions
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def observe(model):
    D = deque()                                # Register where the actions will be stored
    # FIRST STEP: Knowing what each action does (Observing)
    observation = env.reset()                     # Game begins
    obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs 
    state = np.stack((obs, obs), axis=1)
    done = False
    for t in range(OBSERVETIME):
        env.render()
        if np.random.rand() <= EPSILON:
            action = np.random.randint(0, env.action_space.n, size=1)[0]
        else:
            Q = model.predict(state)          # Q-values predictions
            action = np.argmax(Q)             # Move with highest Q-value is the chosen one
        observation_new, reward, done, info = env.step(action)     # See state of the game, reward... after performing the action
        obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)     # Update the input with the new state of the game
        D.append((state, action, reward, state_new, done))         # 'Remember' action and consequence
        state = state_new         # Update state
        if done:
            env.reset()           # Restart game if it's finished
            obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs 
            state = np.stack((obs, obs), axis=1)
    print('Observing Finished')
    return D, state

def learn(model, register, state):
    # SECOND STEP: Learning from the observations (Experience replay)
    minibatch = random.sample(register, MINIBATCH_SIZE)                              # Sample some moves
    inputs_shape = (MINIBATCH_SIZE,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((MINIBATCH_SIZE, env.action_space.n))
    for i in range(0, MINIBATCH_SIZE):
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]        
    # Build Bellman equation for the Q function
        inputs[i:i+1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)
        
        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + GAMMA * np.max(Q_sa)
    # Train network to output the Q function
        model.train_on_batch(inputs, targets)
    print('Learning Finished')

def play(model):
    # THIRD STEP: Play!
    observation = env.reset()
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)
    done = False
    tot_reward = 0.0
    #input("Press [Enter] key to play... !")
    while not done:
        env.render()                    # Uncomment to see game running
        Q = model.predict(state)        
        action = np.argmax(Q)         
        observation, reward, done, info = env.step(action)
        obs = np.expand_dims(observation, axis=0)
        state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)    
        tot_reward += reward
    print('Game ended! Total reward: {}'.format(reward))

def main():
    model = create_model()
    register, state = observe(model)
    learn(model, register, state)
    play(model)


if __name__ == "__main__":
    main()
