import highway_env
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

def visualize_model(model, env, video_folder):
    # Create and wrap the environment

    env = RecordVideo(env, video_folder=video_folder,
                      episode_trigger=lambda e: True)  # record all episodes
    
    # Provide the video recorder to the wrapped environment
    env.unwrapped.set_record_video_wrapper(env)
    
    # Record a video
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        env.render()
    
    env.close()

if __name__ == "__main__":
    from stable_baselines3 import DQN
    
    env = gym.make('highway-fast-v0', render_mode='rgb_array')
    model = DQN.load("highway_fast_v0_dqn_meta_action/model")
    visualize_model(model, env, "run_RL_model")