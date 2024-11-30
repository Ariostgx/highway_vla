import sys
sys.path.append('/u/shuhan/projects/vla')

# %%
from src.models.vlas.cont_obs_token_action_cot_unified_token import ContObsTokenActionCOTVLAUnifiedToken
from src.auto_labeling.highway_env.lane_change import LaneChangeTaskSpec
from transformers import AutoModelForCausalLM, AutoTokenizer

llm_model = 'HuggingFaceTB/SmolLM2-135M-Instruct'

llm_backbone = AutoModelForCausalLM.from_pretrained(llm_model)
tokenizer = AutoTokenizer.from_pretrained(llm_model)

loss_weight = {"action": 1.0, "obs": 0.0, 'reconst': 1.0, "cot": 1.0, "separator": 1.0, "rollout_stop": 1.0}
cot_mode = 'all'
cot_cfg = {'lanes_count': 5, 'max_hop': 4, 'cot_index_mode': 'both'}

if llm_model == 'gpt2':
  hidden_dim = 768
elif llm_model == 'HuggingFaceTB/SmolLM2-135M-Instruct':
  hidden_dim = 576
else:
  raise ValueError(f'Unknown LLM model: {llm_model}')

obs_dim = 25
num_actions = 5
mlp_layers = 2

task_spec_func = LaneChangeTaskSpec

model = ContObsTokenActionCOTVLAUnifiedToken(llm_backbone, tokenizer, task_spec_func, obs_dim, num_actions, hidden_dim, mlp_layers, loss_weight, cot_mode, cot_cfg, max_obs_len=50)


# %%
import torch
import os

ckpt = os.path.expanduser('~/results/vla/quick_run_cot_unified/start_cot_smolLM/lightning_logs/version_0/checkpoints/test_model.ckpt')

lg_ckpt = torch.load(ckpt, map_location='cpu', weights_only=True)['state_dict']
ori_ckpt = {}
for k, v in lg_ckpt.items():
    if k.startswith('vla.'):
        ori_ckpt[k[4:]] = v

model.load_state_dict(ori_ckpt)

# %%
import gymnasium
import highway_env

lanes_cnt_5_cfg = {
    "lanes_count": 5
}
env = gymnasium.make("highway-fast-v0", render_mode='rgb_array', config=lanes_cnt_5_cfg)
curr_obs, _ = env.reset()

curr_obs = torch.tensor(curr_obs, dtype=torch.float32)

# %%
from IPython.display import HTML
import tqdm
import numpy as np
import gymnasium
import highway_env
from matplotlib import pyplot as plt

from transformers.cache_utils import DynamicCache


observations = []
actions = []
reward_names = ['collision_reward', 'right_lane_reward', 'high_speed_reward', 'on_road_reward']
rewards = {name: [] for name in reward_names}

lanes_cnt_5_cfg = {
    "lanes_count": 5
}
env = gymnasium.make("highway-fast-v0", render_mode='rgb_array', config=lanes_cnt_5_cfg)
curr_obs, _ = env.reset()
curr_obs = torch.tensor(curr_obs, dtype=torch.float32)

past_input_str = ''
past_key_value = DynamicCache()
generate_cfg = {'max_new_tokens': 100}


rollout_length = 3  # Adjust
cot_inference_mode = 'pred'
past_input_embeds = None


for _ in range(rollout_length):
    update_str, update_embeddings = model.inference_step(past_input_embeds, past_input_str, past_key_value, curr_obs, cot_inference_mode, generate_cfg)

    act_index = update_str.index('<Act_')
    act_id = int(update_str[act_index+5:act_index+6])

    past_input_str = past_input_str + update_str
    
    if past_input_embeds is None:
        past_input_embeds = update_embeddings
    else:
        past_input_embeds = torch.cat([past_input_embeds, update_embeddings], dim=1)

    obs, reward, done, truncated, info = env.step(act_id)
    curr_obs = torch.tensor(obs, dtype=torch.float32)

    print(update_str)

    for name in reward_names:
        rewards[name].append(info['rewards'][name])

    if done or truncated:
        if truncated:
            print('rollout successfully finished!')
        else:
            print('rollout failed!')
        break

env.close()

avg_rewards = {name: np.mean(rewards[name]) for name in reward_names}

