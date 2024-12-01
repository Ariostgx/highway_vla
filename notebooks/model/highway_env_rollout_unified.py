import sys
import os
import pickle
import random
import logging
import tqdm
import numpy as np
import torch
import gymnasium
import highway_env
from matplotlib import pyplot as plt
from IPython.display import HTML
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.append('/u/shuhan/projects/vla')

from src.models.vlas.cont_obs_token_action_cot_unified_token import ContObsTokenActionCOTVLAUnifiedToken
from src.auto_labeling.highway_env.lane_change import LaneChangeTaskSpec


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='full_cot_smolLM', help='Name of model to use')
parser.add_argument('--rollout_count', type=int, default=1000, help='Number of rollouts to perform')
parser.add_argument('--cot_inference_mode', type=str, default='never', help='Chain of thought inference mode')

args = parser.parse_args()

model_name = args.model_name 
rollout_count = args.rollout_count
cot_inference_mode = args.cot_inference_mode

exp_name = f'{model_name}_{cot_inference_mode}'
save_dir = '/u/shuhan/projects/vla/data/highway_env/rollout_experiment'

model_paths = {'full_cot_smolLM': '~/results/vla/quick_run_cot_unified/full_cot_smolLM/lightning_logs/version_6/checkpoints/test_model.ckpt',
               'start_cot_smolLM': '~/results/vla/quick_run_cot_unified/start_cot_smolLM/lightning_logs/version_0/checkpoints/test_model.ckpt',
               'never_cot_smolLM': '~/results/vla/quick_run_cot_unified/no_cot_smolLM/lightning_logs/version_1/checkpoints/test_model.ckpt',
               'full_cot_gpt2': '~/results/vla/quick_run_cot_unified/full_cot_gpt_2/lightning_logs/version_2/checkpoints/test_model.ckpt'}

if 'gpt2' in model_name:
  llm_model = 'gpt2'
else:
  llm_model = 'HuggingFaceTB/SmolLM2-135M-Instruct'

llm_backbone = AutoModelForCausalLM.from_pretrained(llm_model)
tokenizer = AutoTokenizer.from_pretrained(llm_model)

loss_weight = {"action": 1.0, "obs": 0.0, 'reconst': 1.0, "cot": 1.0, "separator": 1.0, "rollout_stop": 1.0}
cot_mode = 'start'
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


ckpt = os.path.expanduser(model_paths[model_name])

lg_ckpt = torch.load(ckpt, map_location='cpu', weights_only=True)['state_dict']
ori_ckpt = {}
for k, v in lg_ckpt.items():
    if k.startswith('vla.'):
        ori_ckpt[k[4:]] = v

model.load_state_dict(ori_ckpt)

goal_spec_dataset_path = '/u/shuhan/projects/vla/data/highway_env/lane_change_goal_spec_data.pkl'
with open(goal_spec_dataset_path, 'rb') as f:
  goal_spec_dataset = pickle.load(f)

def get_ego_lane_id(curr_obs):
  lane_cnt = 5
  lane_width = 1.0 / lane_cnt
  abs_y = curr_obs[..., 2].copy()
  abs_y[1:] += abs_y[:1]
  abs_y += lane_width / 2
  lane_ids = (abs_y / lane_width).astype(int)
  ego_lane_id = lane_ids[0]
  return ego_lane_id

def compute_path_score(goal_path: list[int], ego_lane_ids: list[int]):
  # exact match
  exact_match_count = sum(1 for g, e in zip(goal_path, ego_lane_ids) if g == e)
  exact_match_score = exact_match_count / len(goal_path)

  # subset coverage
  sequence_matcher = SequenceMatcher(None, goal_path, ego_lane_ids)
  longest_match_length = sequence_matcher.find_longest_match(0, len(goal_path), 0, len(ego_lane_ids)).size
  subset_coverage = longest_match_length / len(goal_path)

  return exact_match_score, subset_coverage

def rollout_one_episode(model, goal_spec_dataset, cot_inference_mode: str):
    env = gymnasium.make("highway-fast-v0", render_mode='rgb_array', config={"lanes_count": 5})
    curr_obs, _ = env.reset()
    ego_lane_id = get_ego_lane_id(curr_obs)

    sampled_path_info = random.choice(goal_spec_dataset[ego_lane_id])

    goal_spec = sampled_path_info['goal_spec']
    hop_lane_ids = sampled_path_info['hop_lane_ids']

    start_id = hop_lane_ids[0]
    goal_id = hop_lane_ids[-1]

    curr_obs = torch.tensor(curr_obs, dtype=torch.float32)

    max_rollout_length = 30

    ego_lane_ids = [start_id]
    actions = []
    model_failed = False
    rollout_collision = False

    past_input_str = goal_spec
    past_key_value = DynamicCache()
    past_input_embeds = model.llm_backbone.get_input_embeddings()(model.llm_tokenizer(past_input_str, return_tensors='pt').input_ids.to(curr_obs.device))

    generate_cfg = {'max_new_tokens': 100, 'do_sample': False}

    for _ in range(max_rollout_length):
        update_str, update_embeddings = model.inference_step(past_input_embeds, past_input_str, past_key_value, curr_obs, cot_inference_mode, generate_cfg)

        past_input_str = past_input_str + update_str
        
        if past_input_embeds is None:
            past_input_embeds = update_embeddings
        else:
            past_input_embeds = torch.cat([past_input_embeds, update_embeddings], dim=1)

        if '<EndOfRollout>' in update_str:
            print('model called end of rollout!')
            break

        if '<Act_' not in update_str:
            print('no action token in the update string!')
            model_failed = True
            break

        act_index = update_str.index('<Act_')
        act_id = int(update_str[act_index+5:act_index+6])

        obs, reward, has_collision, truncated, info = env.step(act_id)
        ego_lane_id = get_ego_lane_id(obs)

        actions.append(act_id)
        ego_lane_ids.append(ego_lane_id)

        curr_obs = torch.tensor(obs, dtype=torch.float32)

        if truncated:
            print('rollout finished!')
            break

        if has_collision:
            rollout_collision = True
            print('rollout collision!')
            break

    # remove repeating lane ids
    ego_lane_ids = [ego_lane_ids[0]] + [ego_lane_ids[i] for i in range(1, len(ego_lane_ids)) if ego_lane_ids[i] != ego_lane_ids[i-1]]

    token_count = past_input_embeds.shape[1]
    action_count = len(actions)
    reached_goal = (ego_lane_ids[-1] == goal_id) and not (model_failed or rollout_collision) and len(ego_lane_ids) == len(hop_lane_ids)
    exact_match_score, subset_coverage = compute_path_score(hop_lane_ids, ego_lane_ids)

    exceeded_length = max(0, len(ego_lane_ids) - len(hop_lane_ids))

    env.close()

    scores = {'token_count': token_count, 'action_count': action_count, 'exact_match_score': exact_match_score, 'subset_coverage': subset_coverage, 'model_failed': model_failed, 'rollout_collision': rollout_collision, 'reached_goal': reached_goal, 'exceeded_length': exceeded_length}

    return scores, past_input_str

os.makedirs(save_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(save_dir, f'{exp_name}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

all_scores = {'token_count': [], 'action_count': [], 'exact_match_score': [], 'subset_coverage': [], 'model_failed': [], 'rollout_collision': [], 'reached_goal': [], 'exceeded_length': []}
all_past_input_str = []


for rollout_idx in tqdm.tqdm(range(rollout_count)):
    scores, past_input_str = rollout_one_episode(model, goal_spec_dataset, cot_inference_mode)
    for k, v in scores.items():
        all_scores[k].append(v)
    all_past_input_str.append(past_input_str)

    logging.info(f'rollout {rollout_idx} done')

    for k, v in all_scores.items():
        logging.info(f'\t {k}: {np.mean(v)}')

logging.info('final results:')
for k, v in all_scores.items():
    logging.info(f'\t {k}: {np.mean(v)}')