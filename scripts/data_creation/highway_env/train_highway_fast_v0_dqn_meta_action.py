import gymnasium
import highway_env
from stable_baselines3 import DQN

steps = 1e5
saved_path = '/u/shuhan/projects/vla/data/highway_env/highway_fast_v0_dqn_meta_action'
model_path = saved_path + '/model'

env = gymnasium.make("highway-fast-v0", render_mode='rgb_array')
model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log=saved_path)
model.learn(int(steps))
model.save(model_path)
