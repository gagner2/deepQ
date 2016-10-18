# Deep Q network
import gym
import numpy as np
import tensorflow as tf
import math
import random

# HYPERPARMETERS
H = 200
H2 = 200
batch_number = 500
gamma = 0.99
num_between_q_copies = 150
explore_decay = 0.9995
min_explore = 0.01
max_steps = 199    
max_episodes = 1000
memory_size = 100000
learning_rate = 0.01
    

if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    env.monitor.start('training', force=True)

    #Setup tensorflow    
    tf.reset_default_graph()

    #First Q Network
    w1 = tf.Variable(tf.random_uniform([env.observation_space.shape[0],H], -1.0, 1.0))
    b1 = tf.Variable(tf.random_uniform([H], -1.0, 1.0))
    
    w2 = tf.Variable(tf.random_uniform([H, H2], -1.0, 1.0))
    b2 = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))
    
    w3 = tf.Variable(tf.random_uniform([H2, env.action_space.n], -1.0, 1.0))
    b3 = tf.Variable(tf.random_uniform([env.action_space.n], -1.0, 1.0))

    #Second Q Network    
    w1_ = tf.Variable(tf.random_uniform([env.observation_space.shape[0],H], -1.0, 1.0))
    b1_ = tf.Variable(tf.random_uniform([H], -1.0, 1.0))
    
    w2_ = tf.Variable(tf.random_uniform([H,H2], -1.0, 1.0))
    b2_ = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))
    
    w3_ = tf.Variable(tf.random_uniform([H2, env.action_space.n], -1.0, 1.0))
    b3_ = tf.Variable(tf.random_uniform([env.action_space.n], -1.0, 1.0))
    
    #Make assign functions for updating Q prime's weights
    w1_update= w1_.assign(w1)
    b1_update= b1_.assign(b1)
    w2_update= w2_.assign(w2)
    b2_update= b2_.assign(b2)
    w3_update= w3_.assign(w3)
    b3_update= b3_.assign(b3)
   
    all_assigns = [
            w1_update, 
            w2_update, 
            w3_update, 
            b1_update, 
            b2_update, 
            b3_update]


    #build network
    states_ = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]])
    h_1 = tf.nn.relu(tf.matmul(states_, w1) + b1)
    h_2 = tf.nn.relu(tf.matmul(h_1, w2) + b2)
    h_2 = tf.nn.dropout(h_2, .5)
    Q = tf.matmul(h_2, w3) + b3

    h_1_ = tf.nn.relu(tf.matmul(states_, w1_) + b1_)
    h_2_ = tf.nn.relu(tf.matmul(h_1_, w2_) + b2_)
    h_2_ = tf.nn.dropout(h_2_, .5)
    Q_ =  tf.matmul(h_2_, w3_) + b3_

    action_used = tf.placeholder(tf.int32, [None], name="action_masks") 
    action_masks = tf.one_hot(action_used, env.action_space.n)
    filtered_Q = tf.reduce_sum(tf.mul(Q, action_masks), reduction_indices=1) 
    
    #Train Q
    target_q = tf.placeholder(tf.float32, [None, ])
    loss = tf.reduce_mean(tf.square(filtered_Q - target_q))
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
    
    #Set up the environment
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
        sess.run(all_assigns)
        
        ticks = 0
        for episode in xrange(max_episodes):
            state = env.reset()
            
            reward_sum = 0
            
            for step in xrange(max_steps):
                ticks += 1

                if episode % 10 == 0:
                    q, qp = sess.run([Q,Q_], feed_dict={states_: np.array([state])})
                    #print "Q:{}, Q_ {}".format(q[0], qp[0])
                    env.render()

                if explore > random.random():
                    action = env.action_space.sample()
                else:
                    q = sess.run(Q, feed_dict={states_: np.array([state])})[0]
                    action = np.argmax(q)
                explore = max(explore * explore_decay, min_explore)
                
                new_state, reward, done, _ = env.step(action)
                reward_sum += reward

                D.append([state, action, reward, new_state, done])
                if len(D) > memory_size:
                    D.pop(0);
           
                state = new_state

               
                if done: 
                    break
                
                #Training a Batch
                samples = random.sample(D, min(batch_number, len(D)))

                #calculate all next Q's together
                new_states = [ x[3] for x in samples]
                all_q = sess.run(Q_, feed_dict={states_: new_states})

                y_ = []
                state_samples = []
                actions = []
                terminalcount = 0
                for ind, i_sample in enumerate(samples):
                    state_mem, curr_action, reward, new_state, done = i_sample
                    if done:
                        y_.append(reward)
                        terminalcount += 1
                    else:
                        this_q = all_q[ind]
                        maxq = max(this_q)
                        y_.append(reward + (gamma * maxq))

                    state_samples.append(state_mem)

                    actions.append(curr_action);
                sess.run([train], feed_dict={states_: state_samples, target_q: y_, action_used: actions})
                if ticks % num_between_q_copies == 0:
                    sess.run(all_assigns)


            if reward_sum >= 190:    
                print 'Reward for episode %d is %d. Explore is %.4f' %(episode,reward_sum, explore)
		    else:
			    print 'Episode %d' %(episode)
                         
env.monitor.close()