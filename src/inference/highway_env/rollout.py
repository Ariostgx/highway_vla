import torch
import gymnasium
import highway_env
from transformers.cache_utils import DynamicCache
from difflib import SequenceMatcher
import copy

def get_ego_lane_id(curr_obs):
    lane_cnt = 5
    lane_width = 1.0 / lane_cnt
    abs_y = curr_obs[..., 2].copy()
    abs_y[1:] += abs_y[:1]
    abs_y += lane_width / 2
    lane_ids = (abs_y / lane_width).astype(int)
    ego_lane_id = lane_ids[0]
    return ego_lane_id.item()


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
def rollout_one_episode(model, env_state_cache, path_info, use_wm, wm_mode, cot_mode, max_rewind_step):
    device = next(model.parameters()).device

    wm_init_collision_cnt = 0 # initial action has collision
    model_wm_cnt = 0 # model wm used
    model_rewind_cnt = 0 # model decide to rewind
    model_rewind_collision_cnt = 0 # model rewind has collision
    wm_init_collision_model_rewind_cnt = 0 # model collision after rewind

    env = gymnasium.make("highway-fast-v0", render_mode='rgb_array', config={"lanes_count": 5})
    env.reset()
    env.unwrapped.__dict__.update(copy.deepcopy(env_state_cache))
    curr_obs = env.unwrapped.observation_type.observe()
    ego_lane_id = get_ego_lane_id(curr_obs)

    goal_spec = path_info['goal_spec']
    hop_lane_ids = path_info['hop_lane_ids']

    start_id = hop_lane_ids[0]

    curr_obs = torch.tensor(curr_obs, dtype=torch.float32).to(device)

    max_rollout_length = 30

    ego_lane_ids = [start_id]
    actions = []
    model_failed = False
    rollout_collision = False

    past_input_str = goal_spec
    past_input_embeds = model.llm_backbone.get_input_embeddings()(model.llm_tokenizer(past_input_str, return_tensors='pt').input_ids.to(curr_obs.device))
    past_key_values = None  # Initialize KV cache

    generate_cfg = {'max_new_tokens': 100, 'do_sample': False}

    max_rewind_step = max_rewind_step + 1

    for _ in range(max_rollout_length):
        # step 1: obtain initial action prediction
        init_act_str, init_act_embeddings, past_key_values = model.init_action_inference(past_input_embeds, past_input_str, curr_obs, generate_cfg)
        
        if '<EndOfRollout>' in init_act_str:
            break

        if '<Act_' not in init_act_str:
            model_failed = True
            break

        init_act_index = init_act_str.index('<Act_')
        init_act_id = int(init_act_str[init_act_index+5:init_act_index+6])

        past_input_str = past_input_str + init_act_str
        past_input_embeds = torch.cat([past_input_embeds, init_act_embeddings], dim=1)

        # print('After STEP 1, past_input_embeds.shape', past_input_embeds.shape)
        # print('After STEP 1, past_key_values[0][0].shape', past_key_values[0][0].shape)

        # step 2: obtain cot start token, decide whether to use cot or not
        # print('STEP 2')
        cot_token_str, cot_token_embeddings, past_key_values = model.cot_start_inference(past_input_embeds, past_input_str, past_key_values, cot_mode, use_wm)
        if len(cot_token_str) > 0:
            past_input_str = past_input_str + cot_token_str
            past_input_embeds = torch.cat([past_input_embeds, cot_token_embeddings], dim=1)
        
        # print('After STEP 2, past_input_embeds.shape', past_input_embeds.shape)
        # print('After STEP 2, past_key_values[0][0].shape', past_key_values[0][0].shape)

        if '<COMMIT>' in cot_token_str:
            final_act_id = init_act_id
        else:
            for rewind_step in range(max_rewind_step):
                # step 3.1: obtain world model prediction
                # print('STEP 3.1')
                model_wm_cnt += 1
                if wm_mode == 'model':
                    wm_str, wm_embeddings, past_key_values = model.cot_append_wm_embeddings(past_input_embeds, past_input_str, past_key_values, None)
                elif wm_mode == 'env':
                    wm_obs, wm_has_collision = get_wm_obs_from_env(env, init_act_id)
                    wm_obs = torch.tensor(wm_obs, dtype=torch.float32).to(curr_obs.device)
                    wm_str, wm_embeddings, past_key_values = model.cot_append_wm_embeddings(past_input_embeds, past_input_str, past_key_values, wm_obs)
                else:
                    raise ValueError(f'Invalid wm mode: {wm_mode}')
                
                past_input_str = past_input_str + wm_str
                past_input_embeds = torch.cat([past_input_embeds, wm_embeddings], dim=1)

                # print('After STEP 3.1, past_input_embeds.shape', past_input_embeds.shape)
                # print('After STEP 3.1, past_key_values[0][0].shape', past_key_values[0][0].shape)

                # step 3.2: conduct wm reflection with <BOT> and <EOT>
                # print('Before STEP 3.2, past_input_str:', past_input_str)
                # print('STEP 3.2')
                reflect_str, reflect_embeddings, past_key_values = model.cot_commit_inference(past_input_embeds, past_input_str, past_key_values, generate_cfg, ending_token='<EOT>')
                past_input_str = past_input_str + reflect_str
                past_input_embeds = torch.cat([past_input_embeds, reflect_embeddings], dim=1)

                # print('After STEP 3.2, past_input_str:', past_input_str)
                # print('After STEP 3.2, past_input_embeds.shape', past_input_embeds.shape)
                # print('After STEP 3.2, past_key_values[0][0].shape', past_key_values[0][0].shape)

                # step 3.3: conduct cot_end_inference
                # print('STEP 3.3')
                cot_end_str, cot_end_embeddings, past_key_values = model.cot_end_inference(past_input_embeds, past_input_str, past_key_values)
                past_input_str = past_input_str + cot_end_str
                past_input_embeds = torch.cat([past_input_embeds, cot_end_embeddings], dim=1)

                # print('After STEP 3.3, past_input_str:', past_input_str)

                if '<COMMIT>' in cot_end_str:
                    final_act_id = init_act_id
                    break
                elif '<BACKSPACE>' in cot_end_str:
                    _, wm_init_collision = get_wm_obs_from_env(env, init_act_id)
                    wm_init_collision_cnt += int(wm_init_collision)
                    model_rewind_cnt += 1

                    # step 3.4 obtain the new action 
                    # print('STEP 3.4')
                    new_act_str, new_act_embeddings, past_key_values = model.cot_commit_inference(past_input_embeds, past_input_str, past_key_values, generate_cfg, ending_token='<EOA>')
                    past_input_str = past_input_str + new_act_str
                    past_input_embeds = torch.cat([past_input_embeds, new_act_embeddings], dim=1)

                    if '<Act_' not in new_act_str:
                        model_failed = True
                        break
                    
                    init_act_id = int(new_act_str[new_act_str.index('<Act_')+5:new_act_str.index('<Act_')+6])

                    # print('After STEP 3.4:', past_input_str)

                    # step 3.5: append BWM token using proper KV cache handling
                    # print('STEP 3.5')
                    bwm_str = '<BWM>'
                    bwm_embeddings = model.llm_backbone.get_input_embeddings()(model.llm_tokenizer(bwm_str, return_tensors='pt').input_ids.to(curr_obs.device))
                    
                    past_input_str = past_input_str + bwm_str
                    past_input_embeds = torch.cat([past_input_embeds, bwm_embeddings], dim=1)

                    # print('After STEP 3.5:', past_input_str)

                    _, wm_final_collision = get_wm_obs_from_env(env, init_act_id)
                    wm_init_collision_model_rewind_cnt += int(wm_init_collision)
                    model_rewind_collision_cnt += int(wm_final_collision)
                    
        # step 5: take action
        obs, _, has_collision, truncated, _ = env.step(final_act_id)
        ego_lane_id = get_ego_lane_id(obs)
        
        actions.append(final_act_id)
        ego_lane_ids.append(ego_lane_id)

        curr_obs = torch.tensor(obs, dtype=torch.float32).to(device)

        if truncated:
            break

        if has_collision:
            rollout_collision = True
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

    result = {
        'exact_match_score': exact_match_score,
        'subset_coverage': subset_coverage,
        'rollout_collision': rollout_collision,
        'model_failed': model_failed,
        'action_count': action_count,
        'token_count': token_count,
        'reached_goal': reached_goal,
        'exceeded_length': exceeded_length
    }

    for k, v in cot_stats.items():
        result[k] = v

    return result