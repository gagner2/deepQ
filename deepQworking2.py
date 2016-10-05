# Deep Q network

import gym
import numpy as np
import tensorflow as tf
import math
import random

# HYPERPARMETERS
H = 100
H2 = 100
batch = 100
gamma = 0.999
num_between_q_copies = 100
explore_decay = 0.999
min_explore = 0.1
max_steps = 200    
max_episodes = 2000
memory_size = 100000
learning_rate = 1e-3
    
if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    env.monitor.start('training_dir', force=True)
    
	#Setup tensorflow    
    tf.reset_default_graph()

    w1 = tf.Variable(tf.random_uniform([env.observation_space.shape[0],H], -.10, .10))
    bias1 = tf.Variable(tf.random_uniform([H], -.10, .10))
    
    w2 = tf.Variable(tf.random_uniform([H, H2], -.10, .10))
    bias2 = tf.Variable(tf.random_uniform([H2], -.10, .10))
    
    w3 = tf.Variable(tf.random_uniform([H2, env.action_space.n], -.10, .10))
    bias3 = tf.Variable(tf.random_uniform([env.action_space.n], -.10, .10))
    
    w1_prime = tf.Variable(tf.random_uniform([env.observation_space.shape[0],H], -1.0, 1.0))
    bias1_prime = tf.Variable(tf.random_uniform([H], -1.0, 1.0))
    
    w2_prime = tf.Variable(tf.random_uniform([H,H2], -1.0, 1.0))
    bias2_prime = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))
    
    w3_prime = tf.Variable(tf.random_uniform([H2, env.action_space.n], -.010, .010))
    bias3_prime = tf.Variable(tf.random_uniform([env.action_space.n], -.010, .010))
    
    #Make assign functions for updating Q prime's weights
    w1_prime_update= w1_prime.assign(w1)
    bias1_prime_update= bias1_prime.assign(bias1)
    w2_prime_update= w2_prime.assign(w2)
    bias2_prime_update= bias2_prime.assign(bias2)
    w3_prime_update= w3_prime.assign(w3)
    bias3_prime_update= bias3_prime.assign(bias3)
   
    all_assigns = [
            w1_prime_update, 
            w2_prime_update, 
            w3_prime_update, 
            bias1_prime_update, 
            bias2_prime_update, 
            bias3_prime_update]

    #build network
    states_placeholder = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]])
    hidden_1 = tf.nn.relu(tf.matmul(states_placeholder, w1) + bias1)
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + bias2)
    hidden_2 = tf.nn.dropout(hidden_2, .5)
    Q = tf.matmul(hidden_2, w3) + bias3

    hidden_1_prime = tf.nn.relu(tf.matmul(states_placeholder, w1_prime) + bias1_prime)
    hidden_2_prime = tf.nn.relu(tf.matmul(hidden_1_prime, w2_prime) + bias2_prime)
    hidden_2_prime = tf.nn.dropout(hidden_2_prime, .5)
    Q_prime =  tf.matmul(hidden_2_prime, w3_prime) + bias3_prime

    action_used_placeholder = tf.placeholder(tf.int32, [None], name="action_masks") 
    action_masks = tf.one_hot(action_used_placeholder, env.action_space.n)
    filtered_Q = tf.reduce_sum(tf.mul(Q, action_masks), reduction_indices=1) 
    
    #we need to train Q
    target_q_placeholder = tf.placeholder(tf.float32, [None]) # This holds all the rewards that are real/enhanced with Qprime
    loss = tf.reduce_sum(tf.square(filtered_Q - target_q_placeholder))
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
    
    #Setting up the enviroment

    D = []
    explore = 1.0
    
    rewardList = []
    past_actions = []
    
    episode_number = 0
    episode_reward = 0
    reward_sum = 0
    
    init = tf.initialize_all_variables()
   
    with tf.Session() as sess:
        sess.run(init)
        #Copy Q over to Q_prime
        sess.run(all_assigns)
   
        ticks = 0
        for episode in xrange(max_episodes):
            state = env.reset()
            reward_sum = 0
            
            for step in xrange(max_steps):
                ticks += 1
                if episode % 10 == 0:
                    q, qp = sess.run([Q,Q_prime], feed_dict={states_placeholder: np.array([state])})
                    print "Q:{}, Q_ {}".format(q[0], qp[0])
                    env.render()

                if explore > random.random():
                    action = env.action_space.sample()
                else:
                    #get action from policy
                    q = sess.run(Q, feed_dict={states_placeholder: np.array([state])})[0]
                    action = np.argmax(q)
                    #print action
                explore = max(explore * explore_decay, min_explore)
                
                new_state, reward, done, _ = env.step(action)
                reward_sum += reward
                #print reward

                D.append([state, action, reward, new_state, done])
                if len(D) > memory_size:
                    D.pop(0);
           
                state = new_state

               
                if done: 
                    break
                
                #Training a Batch
                samples = random.sample(D, min(batch, len(D)))

                #print samples

                #calculate all next Q's together for speed
                new_state = [ x[3] for x in samples]
                all_q_prime = sess.run(Q_prime, feed_dict={states_placeholder: new_state})

                y_ = []
                state_samples = []
                actions = []
                for ind, i_sample in enumerate(samples):
                    state_mem, curr_action, reward, new_state, done = i_sample
                    if done:
                        y_.append(reward)
                    else:
                        #this_q_prime = sess.run(Q_prime, feed_dict={states_placeholder: [new_state]})[0]
                        this_q_prime = all_q_prime[ind]
                        maxq = max(this_q_prime)
                        y_.append(reward + (gamma * maxq))

                    state_samples.append(state_mem)

                    actions.append(curr_action);
                sess.run([train], feed_dict={states_placeholder: state_samples, target_q_placeholder: y_, action_used_placeholder: actions})
                if ticks % num_between_q_copies == 0:
                    sess.run(all_assigns)
                
            print 'Reward for episode %f is %f. Explore is %f' %(episode,reward_sum, explore)
                
                
env.monitor.close()
gym.upload('/home/gagner2/gym/training_dir', api_key='sk_ROAuXzgaRBSMRv9UR1rhJA')