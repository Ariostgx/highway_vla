import sys
import os
import pickle
import random
import logging
import tqdm
import numpy as np
import torch
import gymnasium
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from difflib import SequenceMatcher
import copy

sys.path.append('/u/shuhan/projects/vla')

from src.models.vlas.cont_obs_token_action_cot_unified_token_collision import ContObsTokenActionCOTVLAUnifiedTokenCollision
from src.auto_labeling.highway_env.lane_change import LaneChangeTaskSpecCollision

# Argument Parsing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='full_cot_smolLM', help='Name of model to use')
parser.add_argument('--rollout_count', type=int, default=1000, help='Number of rollouts to perform')
parser.add_argument('--use_wm', action='store_true', help='Use World Model for collision prediction')
parser.add_argument('--wm_mode', type=str, default='model', help='World Model mode')
parser.add_argument('--cot_mode', type=str, default='pred', help='Chain of thought inference mode') # pred, always, never

args = parser.parse_args()

model_name = args.model_name
rollout_count = args.rollout_count
use_wm = args.use_wm
wm_mode = args.wm_mode
cot_mode = args.cot_mode

loss_weight = {
    "action": 1.0,
    "obs": 0.0,
    'reconst': 1.0,
    "cot": 1.0,
    "separator": 1.0,
    "rollout_stop": 1.0,
    "wm": 1.0
}

cot_cfg = {
    'lanes_count': 5,
    'max_hop': 4,
    'cot_index_mode': 'both',
    'action_sample_mode': 'future',
    'safe_reflect_rate': 0.3,
    'collide_reflect_rate': 0.8,
    'collide_rewind_rate': 0.8,
    'max_rewind_step': 1
}


ckpt_dicts = {
   'with_wm_cr_0.8_re_0.2_sr_0.2': '~/results/vla/quick_run_cot_unified_collision/with_wm_cr_0.8_re_0.2_sr_0.2/lightning_logs/version_0/checkpoints/test_model.ckpt',
   'with_wm_cr_0.2_re_0.5_sr_0.2': '~/results/vla/quick_run_cot_unified_collision/with_wm_cr_0.2_re_0.5_sr_0.2/lightning_logs/version_2/checkpoints/test_model.ckpt',
   "with_wm_cr_0.8_re_0.8_sr_0.2": "~/results/vla/quick_run_cot_unified_collision/with_wm_cr_0.8_re_0.8_sr_0.2/lightning_logs/version_1/checkpoints/test_model.ckpt",
   'no_wm_cr_0.8_re_0.8_sr_0.2': '~/results/vla/quick_run_cot_unified_collision/no_wm_cr_0.8_re_0.8_sr_0.2/lightning_logs/version_2/checkpoints/test_model.ckpt',
   "with_wm_cr_0.8_re_0.8_sr_0.2_mask_collision_act": "~/results/vla/quick_run_cot_unified_collision/with_wm_cr_0.8_re_0.8_sr_0.2_mask_collision_act/lightning_logs/version_0/checkpoints/test_model.ckpt",
}

ckpt = ckpt_dicts[model_name]
ckpt = os.path.expanduser(ckpt)
# Utility Functions
def get_ego_lane_id(curr_obs):
    lane_cnt = 5
    lane_width = 1.0 / lane_cnt
    abs_y = curr_obs[..., 2].copy()
    abs_y[1:] += abs_y[:1]
    abs_y += lane_width / 2
    lane_ids = (abs_y / lane_width).astype(int)
    ego_lane_id = lane_ids[0]
    return ego_lane_id


def compute_path_score(goal_path, ego_lane_ids):
    exact_match_count = sum(1 for g, e in zip(goal_path, ego_lane_ids) if g == e)
    exact_match_score = exact_match_count / len(goal_path)
    sequence_matcher = SequenceMatcher(None, goal_path, ego_lane_ids)
    longest_match_length = sequence_matcher.find_longest_match(0, len(goal_path), 0, len(ego_lane_ids)).size
    subset_coverage = longest_match_length / len(goal_path)
    return exact_match_score, subset_coverage


def get_wm_obs_from_env(env, action_id):
    start_env_state = copy.deepcopy(env.__dict__)
    wm_env = gymnasium.make("highway-fast-v0", render_mode='rgb_array', config={"lanes_count": 5})
    _, _ = wm_env.reset()
    wm_env.__dict__.update(start_env_state)
    wm_obs, _, has_collision, _, _ = wm_env.step(action_id)
    return wm_obs, has_collision


# Rollout Function
def rollout_one_episode(model, goal_spec_dataset, use_wm, wm_mode, cot_mode):
    device = next(model.parameters()).device

    wm_init_collision_cnt = 0 # initial action has collision
    model_wm_cnt = 0 # model wm used
    model_rewind_cnt = 0 # model decide to rewind
    model_rewind_collision_cnt = 0 # model rewind has collision
    wm_init_collision_model_rewind_cnt = 0 # model collision after rewind

    env = gymnasium.make("highway-fast-v0", render_mode='rgb_array', config={"lanes_count": 5})
    curr_obs, _ = env.reset()
    ego_lane_id = get_ego_lane_id(curr_obs)

    sampled_path_info = random.choice(goal_spec_dataset[ego_lane_id])

    goal_spec = sampled_path_info['goal_spec']
    hop_lane_ids = sampled_path_info['hop_lane_ids']

    start_id = hop_lane_ids[0]

    curr_obs = torch.tensor(curr_obs, dtype=torch.float32).to(device)

    max_rollout_length = 30

    ego_lane_ids = [start_id]
    actions = []
    model_failed = False
    rollout_collision = False

    past_input_str = goal_spec

    print(goal_spec)

    past_key_value = DynamicCache()
    past_input_embeds = model.llm_backbone.get_input_embeddings()(model.llm_tokenizer(past_input_str, return_tensors='pt').input_ids.to(curr_obs.device))

    generate_cfg = {'max_new_tokens': 100, 'do_sample': False}

    for _ in range(max_rollout_length):
      # step 1: obtain initial action prediction
      init_act_str, init_act_embeddings = model.init_action_inference(past_input_embeds, past_input_str, curr_obs, generate_cfg)
      
      # print('\tinit_act_str:', init_act_str)

      if '<EndOfRollout>' in init_act_str:
        print('model called end of rollout!')
        break

      if '<Act_' not in init_act_str:
        # print('\tno action token in the initial action inference string!')
        model_failed = True
        break

      init_act_index = init_act_str.index('<Act_')
      init_act_id = int(init_act_str[init_act_index+5:init_act_index+6])

      past_input_str = past_input_str + init_act_str
      past_input_embeds = torch.cat([past_input_embeds, init_act_embeddings], dim=1)

      # step 2: obtain cot start token, decide whether to use cot or not
      cot_token_str, cot_token_embeddings = model.cot_start_inference(past_input_embeds, past_input_str, cot_mode, use_wm)
      
      if len(cot_token_str) > 0:
        past_input_str = past_input_str + cot_token_str
        past_input_embeds = torch.cat([past_input_embeds, cot_token_embeddings], dim=1)

      # print('\tcot_token_str:', cot_token_str)

      if '<COMMIT>' in cot_token_str:
        final_act_id = init_act_id
      else:
        # step 3: obtain world model prediction
        if "<BWM>" in cot_token_str and use_wm:
          model_wm_cnt += 1
          if wm_mode == 'model':
            wm_str, wm_embeddings = model.cot_append_wm_embeddings(past_input_embeds, past_input_str, None)
          elif wm_mode == 'env':
            wm_obs, wm_has_collision = get_wm_obs_from_env(env, init_act_id)
            wm_obs = torch.tensor(wm_obs, dtype=torch.float32).to(curr_obs.device)
            wm_str, wm_embeddings = model.cot_append_wm_embeddings(past_input_embeds, past_input_str, wm_obs)
          
          past_input_str = past_input_str + wm_str
          past_input_embeds = torch.cat([past_input_embeds, wm_embeddings], dim=1)
        
        # step 4: obtain cot commit token
        commit_str, commit_embeddings = model.cot_commit_inference(past_input_embeds, past_input_str, generate_cfg)
        past_input_str = past_input_str + commit_str
        past_input_embeds = torch.cat([past_input_embeds, commit_embeddings], dim=1)

        # print('\tcommit_str:', commit_str)

        _, wm_init_collision = get_wm_obs_from_env(env, init_act_id)
        wm_init_collision_cnt += int(wm_init_collision)

        if '<COMMIT>' not in commit_str:
          # print('\tcot commit token is not <COMMIT>!')
          model_failed = True
          break
        elif '<BACKSPACE>' in commit_str and '<Act_' in commit_str:
          # rewind and update action
          # print('\trewind and update action!')
          model_rewind_cnt += 1
          final_act_id = int(commit_str[commit_str.index('<Act_')+5:commit_str.index('<Act_')+6])
          
          _, wm_final_collision = get_wm_obs_from_env(env, final_act_id)
          wm_init_collision_model_rewind_cnt += int(wm_init_collision)
          model_rewind_collision_cnt += int(wm_final_collision)
        else:
          # print('\tsafe, continue to use the initial action!')
          final_act_id = init_act_id
          
      # step 5: take action
      obs, reward, has_collision, truncated, info = env.step(final_act_id)
      ego_lane_id = get_ego_lane_id(obs)
      
      # print(f'step: {len(actions)}, action: {final_act_id}, ego_lane_id: {ego_lane_id}')

      actions.append(final_act_id)
      ego_lane_ids.append(ego_lane_id)

      curr_obs = torch.tensor(obs, dtype=torch.float32).to(device)

      if truncated:
          # print('rollout finished!')
          break

      if has_collision:
          rollout_collision = True
          print('rollout collision!')
          break

    cot_stats = {}

    cot_stats['collision_detect_recall'] = (wm_init_collision_model_rewind_cnt / wm_init_collision_cnt) if wm_init_collision_cnt > 0 else None
    cot_stats['rewind_precision'] = (wm_init_collision_model_rewind_cnt / model_rewind_cnt) if model_rewind_cnt > 0 else None
    cot_stats['rewind_collision_avoid_rate'] = 1 - (model_rewind_collision_cnt / model_rewind_cnt) if model_rewind_cnt > 0 else None
    cot_stats['model_rewind_ratio'] = (model_rewind_cnt / model_wm_cnt) if model_wm_cnt > 0 else None
    
    # remove repeating lane ids
    ego_lane_ids = [ego_lane_ids[0]] + [ego_lane_ids[i] for i in range(1, len(ego_lane_ids)) if ego_lane_ids[i] != ego_lane_ids[i-1]]

    token_count = past_input_embeds.shape[1]
    action_count = len(actions)
    reached_goal = (ego_lane_ids[-1] == hop_lane_ids[-1]) and not (model_failed or rollout_collision) and len(ego_lane_ids) == len(hop_lane_ids)
    exact_match_score, subset_coverage = compute_path_score(hop_lane_ids, ego_lane_ids)

    exceeded_length = max(0, len(ego_lane_ids) - len(hop_lane_ids))

    print(ego_lane_ids, hop_lane_ids)

    return {
        'exact_match_score': exact_match_score,
        'subset_coverage': subset_coverage,
        'rollout_collision': rollout_collision,
        'model_failed': model_failed,
        'action_count': action_count,
        'token_count': token_count,
        'cot_stats': cot_stats,
        'reached_goal': reached_goal,
        'exceeded_length': exceeded_length
    }

exp_name = f'{model_name}_wm_{wm_mode}_cot_{cot_mode}'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(exp_name)
print(device)

# # Model Loading
llm_model = 'HuggingFaceTB/SmolLM2-135M-Instruct'
llm_backbone = AutoModelForCausalLM.from_pretrained(llm_model)
tokenizer = AutoTokenizer.from_pretrained(llm_model)

hidden_dim = 576 if llm_model == 'HuggingFaceTB/SmolLM2-135M-Instruct' else 768
obs_dim = 25
num_actions = 5
mlp_layers = 2
task_spec_func = LaneChangeTaskSpecCollision

model = ContObsTokenActionCOTVLAUnifiedTokenCollision(
    llm_backbone, tokenizer, task_spec_func, obs_dim, num_actions,
    hidden_dim, mlp_layers, loss_weight, "all", cot_cfg, max_obs_len=50, use_wm=use_wm
)
lg_ckpt = torch.load(ckpt, map_location='cpu', weights_only=True)['state_dict']
ori_ckpt = {k[4:]: v for k, v in lg_ckpt.items() if k.startswith('vla.')}
model.load_state_dict(ori_ckpt)

model.to(device)

# Load Goal Specification Dataset
goal_spec_dataset_path = '/u/shuhan/projects/vla/data/highway_env/lane_change_goal_spec_data.pkl'
with open(goal_spec_dataset_path, 'rb') as f:
    goal_spec_dataset = pickle.load(f)

# Run Rollouts
save_dir = '/u/shuhan/projects/vla/data/highway_env/rollout_experiment_collision'
os.makedirs(save_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(save_dir, f'{exp_name}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


all_scores = {'token_count': [], 'action_count': [], 'exact_match_score': [], 'subset_coverage': [], 'model_failed': [], 'rollout_collision': [], 'reached_goal': [], 'exceeded_length': [], 'collision_detect_recall': [], 'rewind_precision': [], 'rewind_collision_avoid_rate': [], 'model_rewind_ratio': []}

for rollout_idx in tqdm.tqdm(range(rollout_count)):
    with torch.no_grad():
      scores = rollout_one_episode(model, goal_spec_dataset, use_wm, wm_mode, cot_mode)

    for k, v in scores.items():
      if k == 'cot_stats':
        for kk, vv in v.items():
          if vv is not None:
            all_scores[kk].append(vv)
      else:
        all_scores[k].append(v)
    
    logging.info(f'rollout {rollout_idx} done')
    for k, v in all_scores.items():
      if len(v) > 0:
        logging.info(f"\t{k}: {np.mean(v)}")
      else:
        logging.info(f"\t{k}: nan")

logging.info('final results:')
for k, v in all_scores.items():
  if len(v) > 0:
    logging.info(f"\t{k}: {np.mean(v)}")
  else:
    logging.info(f"\t{k}: nan")