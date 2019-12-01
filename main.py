import gym
import logging
import tensorflow as tf
import numpy as np
from network import *

def convert_gym_state(state):
    state = tf.image.convert_image_dtype(state,tf.float32)
    state = tf.expand_dims(state,0)
    return state

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # openai-gym config of this game
    env = gym.make('Freeway-v0')
    # observation_space: Box(210,160,3)
    NUM_STATES = env.observation_space.shape
    # action_space: Discrete(3)
    NUM_ACTIONS = env.action_space.n
    NUM_EPOSIDES = 4000
    NUM_BATCHES = 32
    INITIAL_EPSILON = 0.4
    FINAL_EPSILON = 0.05
    EPSILON_DECAY = 1000000
    TRAINING_CYCLE = 2000
    TARGET_UPDATE_CYCLE = 100
    epsilon = INITIAL_EPSILON

    outdir = './results'
    env = gym.wrappers.Monitor(env,directory=outdir,force=True)
    network = DeepQNetwork(NUM_STATES,NUM_BATCHES,NUM_ACTIONS,TRAINING_CYCLE,TARGET_UPDATE_CYCLE,False)
    for episode in range(NUM_EPOSIDES):
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY
        
        episode_reward = 0
        state = env.reset()
        state = convert_gym_state(state)
        t = 0
        while True:
            # env.render()
            if np.random.uniform() < epsilon:
                action = np.random.randint(0,NUM_ACTIONS)
            else:
                tmp = network.evaluation_network(state)
                action = tf.argmax(tmp[0])
            next_state, reward, done, _ = env.step(action)
            next_state = convert_gym_state(next_state)
            network.append_experience({'state':state,
            'action':[action],'reward':[reward],'next_state': next_state})
            episode_reward += reward

            if network.training_counter >= network.training_cycle:
                network.train()
                network.delete_experience()
            state = next_state
            if done:
                logging.info('Episode {} finished after {} timesteps, total rewards {}'.format(episode, t+1, episode_reward))
                break
            t += 1
        if episode % 5 == 0:
            network.save_weights("rl.h5")
    env.close()