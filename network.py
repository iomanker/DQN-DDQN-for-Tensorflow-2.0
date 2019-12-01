import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.losses import *
from tensorflow.keras import activations

class DeepQNetwork(Model):
    def __init__(self,img_size,batch_size,num_actions,
                 training_cycle,target_update_cycle,enable_DDQN):
        super(DeepQNetwork,self).__init__()
        self.batch_size = batch_size
        self.evaluation_network = self.build_model(img_size,num_actions)
        self.target_network = self.build_model(img_size,num_actions)
        self.opt = tf.keras.optimizers.Adam(0.01)
        # Discount factor
        self.gamma = 0.90
        self.loss_function = MeanSquaredError()

        self.training_counter = 0
        self.training_cycle = training_cycle
        self.target_update_counter = 0
        self.target_update_cycle = target_update_cycle
        self.memroies_nameList = ['state','action','reward','next_state']
        self.memories_dict = {}
        for itemname in self.memroies_nameList:
            self.memories_dict[itemname] = None

        self.enable_DDQN = enable_DDQN
    def build_model(self,img_size,num_actions):
        model = tf.keras.Sequential([\
            Conv2D(32,8,8,input_shape=img_size),
            Activation('relu'),
            Conv2D(64,4,4),
            Activation('relu'),
            Conv2D(128,2,2),
            Activation('relu'),
            Flatten(),
            Dense(512),
            Activation('relu'),
            Dense(num_actions)])
        return model

    def train(self):
        # DQN - Experience Replay for Mini-batch
        random_select = np.random.choice(self.training_cycle,self.batch_size)
        states = self.memories_dict["state"][random_select]
        actions = self.memories_dict["action"][random_select]
        rewards = self.memories_dict["reward"][random_select]
        nextStates = self.memories_dict["next_state"][random_select]
        with tf.GradientTape() as tape:
            q_eval_arr = self.evaluation_network(states)
            q_eval = tf.reduce_max(q_eval_arr,axis=1)
            print("q_eval: {}".format(q_eval))
            if self.enable_DDQN == True:
                # Double Deep Q-Network
                q_values = self.evaluation_network(nextStates)
                q_values_actions = tf.argmax(q_values,axis=1)
                target_q_values = self.target_network(nextStates)
                # discount_factor = target_q_values[range(self.batch_size),q_values_actions]
                indice = tf.stack([range(self.batch_size),q_values_actions],axis=1)
                discount_factor = tf.gather_nd(target_q_values,indice)
            else:
                # Deep Q-Network
                target_q_values = self.target_network(nextStates)
                discount_factor = tf.reduce_max(target_q_values,axis=1)
            
             # Q function
            q_target = rewards + self.gamma * discount_factor
            print("q_target: {}".format(q_target))
            loss = self.loss_function(q_eval,q_target)
        
        gradients_of_network = tape.gradient(loss,self.evaluation_network.trainable_variables)
        self.opt.apply_gradients(zip(gradients_of_network, self.evaluation_network.trainable_variables))
        self.target_update_counter += 1
        # DQN - Frozen update
        if self.target_update_counter % self.target_update_cycle == 0:
            self.target_network.set_weights(self.evaluation_network.get_weights())
        
    def call(self):
        return
        
    def append_experience(self,dict):
        tc = self.training_counter
        for itemname in self.memroies_nameList:
            if self.memories_dict[itemname] is None:
                self.memories_dict[itemname] = dict[itemname]
            else:
                self.memories_dict[itemname] = np.append(self.memories_dict[itemname],dict[itemname],axis=0)
        self.training_counter += 1

    def delete_experience(self):
        for itemname in self.memroies_nameList:
            self.memories_dict[itemname] = None
        self.training_counter = 0