import os
from tqdm import tqdm
import gymnasium
import highway_env
import numpy as np
from stable_baselines3 import DQN
from multiprocessing import Pool, cpu_count

lanes_cnt_5_cfg = {
    "lanes_count": 5
}

def single_rollout(args):
    env_id, model_path, rollout_length, rollout_id, save_dir = args

    save_path = os.path.join(save_dir, f'rollout_{rollout_id}.npz')
    if os.path.exists(save_path):
        return
    
    # Create environment
    env = gymnasium.make("highway-fast-v0", render_mode='rgb_array', config=lanes_cnt_5_cfg)
    
    # Load the trained model
    model = DQN.load(model_path, device='cpu')
    
    observations = []
    actions = []
    
    obs, _ = env.reset()
    for _ in range(rollout_length):
        action, _states = model.predict(obs, deterministic=True)
        observations.append(obs)
        actions.append(action)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    
    # Save the rollout data
    np.savez(save_path,
             observations=np.array(observations),
             actions=np.array(actions))
    
    env.close()

def parallel_rollouts(env_id, model_path, num_rollouts, rollout_length, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Prepare arguments for each rollout
    args_list = [(env_id, model_path, rollout_length, i, save_dir) 
                 for i in range(num_rollouts)]
    
    import random
    random.shuffle(args_list)

    # Use 10 CPU cores
    num_processes = 64
    
    # Run rollouts in parallel
    with Pool(num_processes) as p:
        list(tqdm(p.imap(single_rollout, args_list), total=num_rollouts, desc="Rollouts"))
    
if __name__ == "__main__":
    env_id = "highway-fast-v0"
    model_path = "/u/shuhan/projects/vla/data/highway_env/highway_fast_v0_dqn_meta_action_5_lanes/model"
    num_rollouts = 2000000  # Adjust as needed
    rollout_length = 100  # Adjust as needed
    save_dir = "/storage/Datasets/highway_env/highway_fast_v0_dqn_meta_action_5_lanes/rollouts_train"
    
    parallel_rollouts(env_id, model_path, num_rollouts, rollout_length, save_dir)