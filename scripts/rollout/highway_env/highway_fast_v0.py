import torch
import json
import os
import argparse
import sys
import numpy as np
import gymnasium
import highway_env
from tqdm import tqdm

from stable_baselines3 import DQN

sys.path.append('/u/shuhan/projects/vla')



def load_vla_model(ckpt=None):
    from src.models.vlas.cont_obs_token_action import ContObsTokenActionVLA
    from transformers import AutoModelForCausalLM

    llm_backbone = AutoModelForCausalLM.from_pretrained('gpt2')

    hidden_dim = 768
    obs_dim = 25
    num_actions = 5
    mlp_layers = 2

    model = ContObsTokenActionVLA(llm_backbone, obs_dim, num_actions, hidden_dim, mlp_layers)

    # ckpt = os.path.expanduser('~/results/vla/quick_run/action_only/lightning_logs/version_1/checkpoints/ep_68.ckpt')
    if ckpt is None:
        ckpt = os.path.expanduser('~/results/vla/quick_run/action_only/lightning_logs/version_1/checkpoints/ep_140.ckpt')

    lg_ckpt = torch.load(ckpt, map_location='cpu', weights_only=True)['state_dict']
    ori_ckpt = {}
    for k, v in lg_ckpt.items():
        if k.startswith('vla.'):
            ori_ckpt[k[4:]] = v

    model.load_state_dict(ori_ckpt)

    return model

def load_dqn_model():
    model_path = "/u/shuhan/projects/vla/data/highway_env/highway_fast_v0_dqn_meta_action/model"
    model = DQN.load(model_path, device='cpu')
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--num_rollouts', type=int, default=1000)
    parser.add_argument('--ckpt', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model_name == 'vla':
        model = load_vla_model()
    elif args.model_name == 'vla_ckpt':
        model = load_vla_model(args.ckpt)
    elif args.model_name == 'dqn':
        model = load_dqn_model()
    elif args.model_name == 'random':
        print('random action')
    else:
        raise ValueError(f'Invalid model name: {args.model_name}')

    reward_names = ['collision_reward', 'right_lane_reward', 'high_speed_reward', 'on_road_reward']
    result_stats = {name: [] for name in reward_names}
    result_stats['steps'] = []

    for _ in tqdm(range(args.num_rollouts)):
        observations = []
        actions = []
        rewards = {name: [] for name in reward_names}

        env = gymnasium.make('highway-fast-v0', render_mode='rgb_array')
        obs, _ = env.reset()

        observations.append(obs)

        rollout_length = 100  # Adjust

        for _ in range(rollout_length):
            if 'vla' in args.model_name:
                obs_input = torch.tensor(np.stack(observations), dtype=torch.float32)
                past_actions = torch.tensor(actions)
                action = model.predict_action(obs_input, past_actions).argmax().item()
            elif args.model_name == 'dqn':
                action, _ = model.predict(obs, deterministic=True)
            elif args.model_name == 'random':
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)

            observations.append(obs)
            actions.append(action)

            for name in reward_names:
                rewards[name].append(info['rewards'][name])

            if done or truncated:
                break

        env.close()

        avg_rewards = {name: np.mean(rewards[name]) for name in reward_names}
        rollout_steps = len(observations) - 1

        for name in reward_names:
            result_stats[name].append(avg_rewards[name])
        result_stats['steps'].append(rollout_steps)

    save_name = f'{args.model_name}_rollout_stats.json'
    with open(save_name, 'w') as f:
        json.dump(result_stats, f)


if __name__ == '__main__':
    main()