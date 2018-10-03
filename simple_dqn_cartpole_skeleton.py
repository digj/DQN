import gym
import random as rd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class DQN:
    
    REG_FACTOR = 0.1
    REPLAY_MEMORY_SIZE = 1000000          # number of tuples in experience replay  
    EPSILON = 0.5                       # epsilon of epsilon-greedy exploation
    EPSILON_DECAY = 0.99                # exponential decay multiplier for epsilon
    HIDDEN1_SIZE = 256                  # size of hidden layer 1
    HIDDEN2_SIZE = 256                  # size of hidden layer 2
    EPISODES_NUM = 200                 # number of episodes to train on. Ideally shouldn't take longer than 2000
    MAX_STEPS = 1000                     # maximum number of steps in an episode 
    LEARNING_RATE = 0.001              # learning rate and other parameters for SGD/RMSProp/Adam
    MINIBATCH_SIZE = 1000                 # size of minibatch sampled from the experience replay
    DISCOUNT_FACTOR = 0.99               # MDP's gamma
    TARGET_UPDATE_FREQ = 100            # number of steps (not episodes) after which to update the target networks 
    LOG_DIR = './logs'                  # directory wherein logging takes place
    replay_memory = []

    # Create and initialize the environment
    def __init__(self, env):
        self.env = gym.make(env)
        assert len(self.env.observation_space.shape) == 1
        self.input_size = self.env.observation_space.shape[0]       # In case of cartpole, 4 state features
        self.output_size = self.env.action_space.n                  # In case of cartpole, 2 actions (right/left)
    
    # Create the Q-network
    def initialize_network(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        with tf.variable_scope('hidden_layer_1'):
            W_1 = tf.Variable(tf.truncated_normal([self.input_size, self.HIDDEN1_SIZE],stddev = 0.01),name = 'W_1')
            b_1 = tf.Variable(tf.zeros(self.HIDDEN1_SIZE), name = 'b_1')
            h_1 = tf.nn.relu(tf.matmul(self.x,W_1) + b_1)
        with tf.name_scope('hidden_layer_2'):
            W_2 = tf.Variable(tf.truncated_normal([self.HIDDEN2_SIZE, self.HIDDEN1_SIZE],stddev = 0.01),name = 'W_2')
            b_2 = tf.Variable(tf.zeros(self.HIDDEN2_SIZE), name = 'b_2')
            h_2 = tf.nn.relu(tf.add(tf.matmul(h_1,W_2), b_2))
        with tf.name_scope('output_layer'):
            W_3 = tf.Variable(tf.truncated_normal([self.HIDDEN2_SIZE, self.output_size],stddev = 0.01),name = 'W_3')
            b_3 = tf.Variable(tf.zeros(self.output_size), name = 'b_3')
            self.Q = (tf.add(tf.matmul(h_2,W_3), b_3))
        self.params_train = [W_1, b_1, W_2, b_2, W_3, b_3]
        self.Q_t = tf.placeholder(tf.float32, [None])
        self.a = tf.placeholder(tf.float32, [None, self.output_size])
        q_values = tf.reduce_sum(tf.multiply(self.Q, self.a), reduction_indices=[1])
        self.loss = tf.reduce_mean(tf.square(tf.subtract(q_values, self.Q_t)))
        # L2 Reguralization
        for w in [W_1, W_2, W_3]:
            self.loss += self.REG_FACTOR * tf.reduce_sum(tf.square(w))
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.RMSPropOptimizer(self.LEARNING_RATE).minimize(self.loss)
        ############################################################

    def train(self, episodes_num=EPISODES_NUM):
        
        # Initialize summary for TensorBoard                        
        # Summary for TensorBoard
        tf.summary.scalar('loss', self.loss)
        self.summary = tf.summary.merge_all()
        self.sess = tf.Session()
        self.summary_writer = tf.summary.FileWriter(self.LOG_DIR, self.sess.graph)
        self.summary = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        total_steps = 0
        params_target = self.sess.run(self.params_train)
        rew_arr = []
        for episode in range(episodes_num):
            state = self.env.reset()
            episode_steps = 0
            sum_rew = 0
            while True:
                if np.random.uniform() < self.EPSILON:
                    # print("explore")
                    action = rd.choice([0,1])
                else:
                    actions_value = self.sess.run(self.Q, feed_dict={self.x: state.reshape((1,4))})
                    action = np.argmax(actions_value)
                next_state, reward_, done, _ = self.env.step(action)
                sum_rew = sum_rew + reward_
                if done:
                    reward_ = -200
                self.replay_memory.append((state, action, reward_, next_state, done))
                if len(self.replay_memory) > self.REPLAY_MEMORY_SIZE:
                    self.replay_memory.pop(0)
                state = next_state
                self.EPSILON = self.EPSILON * self.EPSILON_DECAY 
                if len(self.replay_memory) >= self.MINIBATCH_SIZE:
                    minibatch = rd.sample(self.replay_memory, self.MINIBATCH_SIZE)
                    next_states = [m[3] for m in minibatch]
                    feed_dict = {self.x: next_states}
                    feed_dict.update(zip(self.params_train, params_target))
                    q_values = self.sess.run(self.Q, feed_dict=feed_dict)
                    max_q_values = q_values.max(axis=1)
                    # Compute target Q values
                    q_tar = np.zeros(self.MINIBATCH_SIZE)
                    target_action_mask = np.zeros((self.MINIBATCH_SIZE, self.output_size), dtype=int)
                    for i in range(self.MINIBATCH_SIZE):
                        _, action, reward, _, terminal = minibatch[i]
                        q_tar[i] = reward
                        if not terminal:
                            q_tar[i] += self.DISCOUNT_FACTOR * max_q_values[i]
                        target_action_mask[i][action] = 1
                    # Gradient descent
                    states = [m[0] for m in minibatch]
                    feed_dict = {
                      self.x: states, 
                      self.Q_t: q_tar,
                      self.a: target_action_mask,
                    }
                    _, summary = self.sess.run([self.train_op, self.summary], feed_dict=feed_dict)
                total_steps += 1
                episode_steps += 1
                if total_steps % self.TARGET_UPDATE_FREQ == 0:
                    params_target = self.sess.run(self.params_train)
                if done or episode_steps == self.MAX_STEPS:
                    break
            rew_arr.append(episode_steps)
            print("Training: Episode = %d, Length = %d, Global step = %d" % (episode, episode_steps, total_steps))
        rew_arr = np.array(rew_arr)
        rew_arr = np.mean(rew_arr.reshape((20,10)),axis = 1)
        plt.plot(range(len(list(rew_arr))),rew_arr)
        plt.show()



    def playPolicy(self):
        
        done = False
        steps = 0
        state = self.env.reset()
        print(state)
        
        # we assume the CartPole task to be solved if the pole remains upright for 200 steps
        while not done and steps < 200:     
            self.env.render()               
            q_vals = self.sess.run(self.Q, feed_dict={self.x: [state]})
            action = q_vals.argmax()
            state, _, done, _ = self.env.step(action)
            steps += 1
        
        return steps


if __name__ == '__main__':

    # Create and initialize the model
    dqn = DQN('CartPole-v0')
    dqn.initialize_network()

    print("\nStarting training...\n")
    dqn.train()
    print("\nFinished training...\nCheck out some demonstrations\n")

    # Visualize the learned behaviour for a few episodes
    results = []
    for i in range(50):
        episode_length = dqn.playPolicy()
        print("Test steps = ", episode_length)
        results.append(episode_length)
    print("Mean steps = ", sum(results) / len(results)) 

    print("\nFinished.")
    print("\nCiao, and hasta la vista...\n")
