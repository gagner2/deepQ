import numpy as np
import gym
import tensorflow as tf

# %% environment setting

env = gym.make('MountainCar-v0')

# Action Space A : discrete(3)
# 0 : decelerate
# 1 : coast
# 2 : accelerate
A = env.action_space
O = env.observation_space

#  Hyper parameters
num_tilings = 10
num_tiling_width = 9
num_actions = A.n
lambda_ = 0.9
alpha = 0.005  # 0.05 / num_tilings
epsilon_b = 0  # behaviour policy epsilon greedy
epsilon_t = 0  # target policy epsilon greedy
gamma_ = 0.999  # 0.999


# %% tile_coding

def tiling_init():
  '''
    initialize tiling condition
    return tile_shape, tilings_offset
  '''
  tile_shape = (O.high - O.low) / (num_tiling_width - 1)

  tilings_offset = np.zeros([num_tilings, 2])
  for i in range(num_tilings):
    tile_offset_x = np.random.uniform(0, tile_shape[0])
    tile_offset_y = np.random.uniform(0, tile_shape[1])
    tilings_offset[i] = [tile_offset_x, tile_offset_y]

  return tile_shape, tilings_offset


def s2f(state_, tile_shape_, tilings_offset_, action_=0):
  '''
    state to features
    consider it as feature function phi(s)
    return rank 1 tensor(vector)
  '''
  pos = state_ - O.low
  f = np.zeros([num_tilings, num_tiling_width * num_tiling_width])
  f_pair = np.zeros([num_tilings, num_tiling_width *
                     num_tiling_width, num_actions])

  for i_tiling in range(num_tilings):
    temp = np.zeros([num_tiling_width, num_tiling_width])
    pos_ = pos - tilings_offset_[i_tiling]
    x_i = 1 + pos_[0] // tile_shape_[0]
    y_i = 1 + pos_[1] // tile_shape_[1]
    temp[int(x_i)][int(y_i)] = 1
    f[i_tiling, :] = np.reshape(
        temp, num_tiling_width * num_tiling_width)
    f_pair[i_tiling, :, action_] = np.reshape(
        temp, num_tiling_width * num_tiling_width)

  f = np.reshape(f, [num_tilings * num_tiling_width * num_tiling_width])
  f_pair = np.reshape(
      f_pair, [num_tilings * num_tiling_width * num_tiling_width, num_actions])
  return f, f_pair


def sa2f(state_, tile_shape_, tilings_offset_, action_):
  '''
    (state, action) pair to features
    consider it as feature function phi(s,a)
    but return rank 2 tensor(matrix) : (features, action), not rank 1 tensor(vector)
  '''
  pos = state_ - O.low
  f = np.zeros([num_tilings, num_tiling_width * num_tiling_width, num_actions])

  for i_tiling in range(num_tilings):
    temp = np.zeros([num_tiling_width, num_tiling_width])
    pos_ = pos - tilings_offset_[i_tiling]
    x_i = 1 + pos_[0] // tile_shape_[0]
    y_i = 1 + pos_[1] // tile_shape_[1]
    temp[int(x_i)][int(y_i)] = 1
    f[i_tiling, :, action_] = np.reshape(
        temp, num_tiling_width * num_tiling_width)

  return np.reshape(f, [num_tilings * num_tiling_width * num_tiling_width, num_actions])


def f2sa(f_, tile_shape_, tilings_offset_):
  '''
    features to original (state, action) pair
  '''
  f = np.reshape(f_, [num_tilings, num_tiling_width,
                      num_tiling_width, num_actions])
  temp = f[:, :, :, :]
  where_one = np.argwhere(temp == 1)
  print(where_one)
  s = [0, 0]
  n = 0
  for i_tiling, x, y, a in where_one:
    x_ = (x - 1 / 2) * tile_shape_[0]
    y_ = (y - 1 / 2) * tile_shape_[1]
    s += [x_, y_] + tilings_offset_[i_tiling] + O.low
    n += 1

  return s / n

# %% functions

def epsilon_greedy(a, b, rand_):
  if rand_ < epsilon_b:
    return a
  else:
    return b

# In[]: tensorflow graph

# observation : (1d-position, velocity)
observation = env.reset()

graph = tf.Graph()
num_component = num_tilings * num_tiling_width * num_tiling_width

with graph.as_default():

  # S(t)
  input_feature_t = tf.placeholder(
      tf.float32, shape=[num_component], name='feature_t')

  feature_pair = tf.placeholder(
      tf.float32, shape=[num_component, num_actions], name='feature_pair')

  # S(t+1)
  input_feature_t_next = tf.placeholder(
      tf.float32, shape=[num_component], name='feature_t_next')

  # a(t)
  input_action = tf.placeholder(tf.int64, shape=[], name='random_action')
  # r(t+1)
  input_reward = tf.placeholder(tf.float32, shape=[], name='reward')

  # theta
  weights = tf.Variable(tf.random_uniform(
      shape=[num_component, num_actions], minval=-0.01, maxval=0.0), name='theta')
  tf.histogram_summary('weights', weights)

  # eligibility trace vector e(t)
  eligibility = tf.Variable(
      tf.zeros(shape=[num_component, num_actions]), name='eligibility')
  tf.histogram_summary('eligibility', eligibility)
  # estimated Q-value

  with tf.name_scope(name='Greedy_Policy'):
    f_t = tf.reshape(input_feature_t, shape=[1, num_component])
    q_t = tf.reshape(tf.matmul(f_t, weights), shape=[num_actions])

    # action A(t)
    a = tf.argmax(q_t, 0, name='ArgMax_Q')

  with tf.name_scope(name='Compute_Q'):
    f_next = tf.reshape(input_feature_t_next, shape=[
                        1, num_component], name='f_t')
    q_next = tf.reshape(tf.matmul(f_next, weights),
                        shape=[num_actions], name='q_t')

  with tf.name_scope(name='delta'):
    delta = input_reward + gamma_ * \
        tf.gather(q_next, tf.argmax(q_next, 0), name='Q_next') - \
        tf.gather(q_t, input_action, name='Q_t')
    with tf.name_scope(name='delta_done'):
      delta_done = input_reward - tf.gather(q_t, input_action, name='Q_t')

  with tf.name_scope(name='update_eligibility'):
    eligibility = tf.assign(eligibility, tf.maximum(
        lambda_ * gamma_ * eligibility, feature_pair), name='eligibility')

  with tf.name_scope(name='theta_update'):
    optimizer = tf.assign_add(
        weights, alpha * delta * eligibility, name='update_theta')
    with tf.name_scope(name='theta_done'):
      optimizer_done = tf.assign_add(
          weights, alpha * delta_done * eligibility, name='done_theta')

  saver = tf.train.Saver()

  merged = tf.merge_all_summaries()

# In[]: Session for episodes
summaries_dir = './summaries/train'
num_episodes = 2000
ts, to = tiling_init()
outdir = '/tmp/MountainCar-v0-expr-1'

with tf.Session(graph=graph) as session:

  if tf.gfile.Exists(summaries_dir):
    print("delete summary")
    tf.gfile.DeleteRecursively(summaries_dir)
  tf.gfile.MakeDirs(summaries_dir)

  train_writer = tf.train.SummaryWriter(summaries_dir, session.graph)

  tf.initialize_all_variables().run()
  env.monitor.start(outdir, force=True)

  for i_episode in range(num_episodes):
    done = False
    t = 0
    observation = env.reset()
    tf.initialize_variables([tf.all_variables()[1]]).run()
    print("start Episodes : ", i_episode)
    for t in range(400):
      action = A.sample()
      f, _ = s2f(observation, ts, to)
      feed_dict = {input_feature_t: f, input_action: action}
      action = session.run(a, feed_dict=feed_dict)
      _, f_pair = s2f(observation, ts, to, action)
      observation, reward, done, _ = env.step(action)
      f_, _ = s2f(observation, ts, to, action)

      feed_dict = {input_feature_t: f, input_feature_t_next: f_,
                   feature_pair: f_pair, input_reward: reward, input_action: action}

      if done:
        _ = session.run(optimizer_done, feed_dict=feed_dict)

        print("Episode finished after {} timesteps".format(t + 1))
        break

      _ = session.run(optimizer, feed_dict=feed_dict)

    summary = session.run(merged, feed_dict={})
    train_writer.add_summary(summary, i_episode)

  save_path = saver.save(session, "./checkpoint/model.ckpt")
  print("Model saved in file: %s" % save_path)

env.monitor.close()