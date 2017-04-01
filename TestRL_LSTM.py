import numpy as np

import keras
from keras.models import Sequential
from keras.layers import LSTM, Activation
import random
import time
import gym

import sys

#this is hot garbage btw lol

env = gym.make('CartPole-v0')

init = 'he_normal'

model = Sequential([
	LSTM(6, batch_input_shape = (1, 1, 5), stateful=True, return_sequences=True, init=init),
	Activation('relu'),
	LSTM(6, stateful=True, return_sequences=True, init=init),
	Activation('relu'),
	LSTM(1, stateful=True, return_sequences=True, init=init),
	#Activation('linear')
])
model.compile('adam', loss='mse')

actions = [[0], [1]]
q = [0, 0, 0, 0]

Trajectories = []


if(len(sys.argv) >= 3):
	if(sys.argv[1] == "-lw"): # command to load weights from file
		model.load_weights(sys.argv[2])	# specify file to load from in next arg

num_games = 1000
gamma = 1

for game_i in range(num_games):
	print("\n\nGame:\t" + str(game_i) + "  of  " + str(num_games) + "\n\n")
	observation = env.reset()
	epsilon = 1 - game_i/500
	T = []
	while True:	#take actions until end of game

		env.render()
	
		for i,a in enumerate(actions):
			a = np.array(a)
			q[i] = model.predict_on_batch(np.concatenate((observation.reshape(1,4),a.reshape(1,1)), axis=1).reshape(1,1,5))
		action = actions[np.argmax(q)-1]
		print(action)

		if(random.random() > epsilon): # epsilon % of the time, take a random action and end the current trajectory
			action = actions[random.randint(0,1)]
			Trajectories.append(T)	# add trajectory to memory
			T = []
		r_action = np.array(action)
		state_action = np.concatenate((observation.reshape(1,4),r_action.reshape(1,1)),axis=1).reshape(1,1,5)

		s_a_r = [state_action]

		print(str(action) + "\n")
		observation, reward, done, info = env.step(action[0])

		reward = np.array(reward).reshape(1,1,1)

		s_a_r.append(reward)
		T.append(s_a_r)

		#update rewards
		for i in range(len(T)):
			if(i != len(T)-1): T[i][1] += reward * gamma**(len(T)-i-1)
			#print(T[i][1])

		# if len(T) > 0:
		# 	model.train_on_batch(state_action.reshape(1,1,5), reward)


		time.sleep(0.025)
		if done:
			#reset internal memory (but not weights)
			model.reset_states()

			#train on random trajectory
			if(len(Trajectories)>0):
				for s_a_r in Trajectories[random.randint(0, len(Trajectories)-1)]:
					print("prediction: \t" + str(model.predict(s_a_r[0])))
					print("actual:   \t" + str(s_a_r[1]))
					model.train_on_batch(s_a_r[0], s_a_r[1])
				model.reset_states()

			Trajectories.append(T)

			break
	

model.save_weights("TestRL_weights")