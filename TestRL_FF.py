import numpy as np

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import random
import time
import gym

import sys

#this is hot garbage btw lol

env = gym.make('CartPole-v0')

model = Sequential([
	Dense(6, batch_input_shape = (None, 6)),
	Activation('relu'),
	Dense(6),
	Activation('relu'),
	Dense(1),
	Activation('linear')
])
model.compile('adam', loss='mse')

actions = [[0,1], [1,0]]
q = [0, 0, 0, 0]

Trajectories = []


if(len(sys.argv) >= 3):
	if(sys.argv[1] == "-lw"): # command to load weights from file
		model.load_weights(sys.argv[2])	# specify file to load from in next arg

num_games = 10000
gamma = 1

for game_i in range(num_games):
	print("\n\nGame:\t" + str(game_i) + "  of  " + str(num_games) + "\n\n")
	observation = env.reset()
	epsilon = 1 - 2 * game_i/1000
	T = []
	while True:
		env.render()
		for i,a in enumerate(actions):
			a = np.array(a)
			q[i] = model.predict(np.concatenate((observation.reshape(1,4),a.reshape(1,2)),axis=1))
		action = actions[np.argmax(q)-1]
		#todo: rethink this. How is it learning from this?..
		if(random.random() > epsilon): 
			action = actions[random.randint(0,1)]
			Trajectories.append(T)
			T = []
		r_action = np.array(action)
		state_action = np.concatenate((observation.reshape(1,4),r_action.reshape(1,2)),axis=1)

		s_a_r = [state_action]

		action = np.argmax(action)
		print(action)
		observation, reward, done, info = env.step(action)

		reward = np.array(reward).reshape(1,1)

		s_a_r.append(reward)
		T.append(s_a_r)
		for i in range(len(T)):
			if(i != len(T)-1): T[i][1] += reward * gamma**(len(T)-i-1)
			#print(T[i][1])

		model.train_on_batch(state_action, reward)

		#train on random trajectory
		if(len(Trajectories)>0):
			for s_a_r in Trajectories[random.randint(0, len(Trajectories)-1)]:
				print("prediction: \t" + str(model.predict(s_a_r[0])))
				print("actual:   \t" + str(s_a_r[1]))
				model.train_on_batch(s_a_r[0], s_a_r[1])

		time.sleep(0.05)
		if done:
			break
	Trajectories.append(T)

model.save_weights("TestRL_weights")