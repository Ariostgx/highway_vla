from locale import currency
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache

from .cont_obs_token_action_cot_unified_token import ContObsTokenActionCOTVLAUnifiedToken
from ..backbones.mlp import MLP
from ..backbones.observation import VectorObservationAutoencoder

from ...auto_labeling.highway_env.lane_change import LaneChangeTaskSpecCollision

class ContObsTokenActionCOTVLAUnifiedTokenCollision(ContObsTokenActionCOTVLAUnifiedToken):
    def __init__(self, llm_backbone: PreTrainedModel, llm_tokenizer: PreTrainedTokenizer, task_spec_func: LaneChangeTaskSpecCollision, obs_dim: int, num_actions: int, hidden_dim: int, mlp_layers: int, loss_weight: dict[str, float] = {"action": 1.0, 'reconst': 1.0, "cot": 1.0, "separator": 1.0, "rollout_stop": 1.0, 'wm': 1.0}, cot_mode: str = "none", cot_cfg: dict = {}, max_obs_len: int = 50, use_wm: bool = False, mask_collision_action: bool = False):
        super().__init__(llm_backbone, llm_tokenizer, task_spec_func, obs_dim, num_actions, hidden_dim, mlp_layers, loss_weight, cot_mode, cot_cfg, max_obs_len)
        self.use_wm = use_wm
        self.mask_collision_action = mask_collision_action

        assert 'safe_reflect_rate' in cot_cfg, "safe_reflect_rate must be specified in cot_cfg"
        assert 'collide_reflect_rate' in cot_cfg, "collide_reflect_rate must be specified in cot_cfg"
        assert 'collide_rewind_rate' in cot_cfg, "collide_rewind_rate must be specified in cot_cfg"

        # define reflection tokens
        special_reflect_tokens = ['<BACKSPACE>', '<COMMIT>']
        if self.use_wm:
            special_reflect_tokens +=  ['<BWM>', '<EWM>']
            special_reflect_tokens += [f'<WM_{i}>' for i in range(max_obs_len)]
        
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': special_reflect_tokens})
        self.llm_backbone.resize_token_embeddings(len(self.llm_tokenizer))

        self.to_train_wm = loss_weight['wm'] > 0 and self.use_wm

        if self.to_train_wm:
            self.wm_head = MLP(hidden_dim, hidden_dim, hidden_dim, mlp_layers)

    def obtain_input_strs(self, batch_data: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        '''
        Obtain the input strings for the LLM that enable reflection.
        Note: only reflection on collision is used in CoT. Not using progress monitoring now.
        Afterwards, needs to:
            1. Replace the observation placeholder tokens with actual observation embeddings.
            2. Replace WM tokens with actual WM embeddings.
        CoT Sample method:
            1. At a safe state, use SAFE CoT with probability safe_reflect_rate.
            2. At a collision state, use SAFE/REWIND CoT with probability collide_reflect_rate.
            3. At a collision state, select the REWIND CoT with probability collide_rewind_rate.
        CoT Templace:
            1. SAFE CoT: 
                <BOO> <Obs_1><EOO><BOA><Act_1><EOA><BWM><WM_1><EWM><BOT> collision safe, commit.<EOT> <COMMIT>
            2. REWIND CoT:
                <BOO> <Obs_1><EOO><BOA><Act_1><EOA><BWM><WM_1><EWM><BOT> collision detected, revise.<EOT> <BACKSPACE> <BOA> <Act_2><EOA> <COMMIT>
            3. No CoT:
                <BOO> <Obs_1><EOO><BOA><Act_1><EOA> <COMMIT>
        Output:
            batch_input_strs: list[str], the input strings for the LLM
            batch_index_data: dict, the index of the goal specification in the input strings
                'goal_spec_index': list[int], the index of the goal specification in the input strings
                'wm_observations': list[torch.Tensor], the WM observations for the WM tokens; index corresponds to the cnt of the WM tokens
        '''
        observations, actions, valid_mask = batch_data[:3]
        cot_rewind_steps, cot_obs, cot_acts, cot_valid_mask = batch_data[3:]
        
        B = observations.shape[0]
        batch_input_strs = []
        batch_goal_spec_index = [] # use to mask the labels for goal-specification text tokens
        batch_cot_wm_observations = []
        batch_collision_action_index = [] # use to mask the labels for collision action tokens

        for bidx in range(B):
            # the input string for the LLM. need to replace the observation placeholder tokens with actual observation strings after encoding
            input_str = "" 
            cot_wm_observations = []
            collision_action_index = []

            valid_mask_b = valid_mask[bidx]
            cot_valid_mask_b = cot_valid_mask[bidx]
            task_spec = self.task_spec_func(observations[bidx][valid_mask_b].cpu().numpy(), actions[bidx][valid_mask_b].cpu().numpy(), 
                                            cot_obs[bidx][cot_valid_mask_b].cpu().numpy(), cot_acts[bidx][cot_valid_mask_b].cpu().numpy(), 
                                            cot_rewind_steps[bidx][cot_valid_mask_b].cpu().numpy(), self.cot_cfg)
            
            goal_spec = task_spec.get_goal_spec()
            cot_prompt = task_spec.get_multi_step_cot_prompt()
            hop_indices = task_spec.get_task_hop_info()['hop_indices']
            
            cot_tidx = task_spec.cot_tidx
            cot_collision_actions = task_spec.cot_collision_actions
            cot_collision_observations = task_spec.cot_observations

            last_hop_idx = hop_indices[-1]

            input_str += goal_spec
            goal_spec_token_num = self.llm_tokenizer(goal_spec, return_tensors="pt").input_ids.shape[1]
            batch_goal_spec_index.append(goal_spec_token_num)
            
            cot_cnt = 0

            for obs_idx in range(last_hop_idx + 1):
                # only add observation + action at the last hop
                if not valid_mask[bidx, obs_idx]:
                    continue
                
                # step 1: always add ground-truth observation
                input_str += f"<BOO><Obs_{obs_idx}><EOO>"

                is_safe_step = obs_idx not in cot_tidx

                if is_safe_step:
                    use_cot = np.random.uniform() < self.cot_cfg['safe_reflect_rate']
                    safe_cot = True
                else:
                    use_cot = np.random.uniform() < self.cot_cfg['collide_reflect_rate']
                    safe_cot = np.random.uniform() > self.cot_cfg['collide_rewind_rate']
                
                safe_act_id = actions[bidx, obs_idx].cpu().long().item()

                use_cot &= valid_mask[bidx, obs_idx+1] == True
                
                if use_cot:
                    if safe_cot:
                        input_str += f"<BOA><Act_{safe_act_id}><EOA>"
                        
                        if self.use_wm:
                            input_str += f"<BWM><WM_{cot_cnt}><EWM>"
                            safe_obs = observations[bidx][valid_mask_b][obs_idx+1]
                            cot_wm_observations.append(safe_obs)
                            # print('safe_obs.shape', safe_obs.shape)
                        
                        input_str += f"<BOT>Safe<EOT>"
                    else:
                        cidx = cot_tidx.index(obs_idx)
                        collide_act_id = int(cot_collision_actions[cidx])

                        # collide observation is the observation at the collision step
                        collide_obs = torch.tensor(cot_collision_observations[cidx], device=observations.device)

                        if len(collide_obs.shape) == 3:
                            collide_obs = collide_obs.squeeze(0)
                        
                        input_str_token_num = self.llm_tokenizer(input_str, return_tensors="pt").input_ids.shape[1]
                        collision_action_index.append(input_str_token_num+1)

                        input_str += f"<BOA><Act_{collide_act_id}><EOA>"

                        if self.use_wm:
                            input_str += f"<BWM><WM_{cot_cnt}><EWM>"
                            cot_wm_observations.append(collide_obs)
                            # print('collide_obs.shape', collide_obs.shape)
                        
                        input_str += f"<BOT>Collision<EOT>"
                        input_str += f"<BACKSPACE>"
                        input_str += f"<BOA><Act_{safe_act_id}><EOA>"

                    cot_cnt += 1
                
                else:
                    # if not using CoT, add the action and directly commit
                    input_str += f"<BOA><Act_{safe_act_id}><EOA>"

                input_str += f"<COMMIT>"

            input_str += f"<BOO><Obs_{last_hop_idx + 1}><EOO><EndOfRollout>"
            batch_input_strs.append(input_str)
            if len(cot_wm_observations) > 0:
                batch_cot_wm_observations.append(torch.stack(cot_wm_observations))
            else:
                batch_cot_wm_observations.append(None)
            batch_collision_action_index.append(collision_action_index)
        
        batch_index_data = {
            'goal_spec_index': batch_goal_spec_index,
            'wm_observations': batch_cot_wm_observations,
            'collision_action_index': batch_collision_action_index
        }

        return batch_input_strs, batch_index_data

    def replace_obs_placeholder_tokens(self, batch_input_ids: torch.Tensor, batch_input_embeds: torch.Tensor, obs_embed: torch.Tensor, batch_index_data: dict):
        batch_input_embeds, batch_obs_data = super().replace_obs_placeholder_tokens(batch_input_ids, batch_input_embeds, obs_embed, batch_index_data)

        if not self.use_wm:
            return batch_input_embeds, batch_obs_data

        B, T = batch_input_ids.shape[:2]

        wm_token_bidx = []
        wm_token_tidx = []
        wm_target_embeds = []
        batch_obs_mask = batch_obs_data['obs_mask']

        # replace the WM placeholder tokens with actual WM embeddings
        for bidx in range(B):
            if batch_index_data['wm_observations'][bidx] is not None:
                wm_observations = batch_index_data['wm_observations'][bidx]
                
                for wm_idx in range(wm_observations.shape[0]):
                    wm_token_id = self.llm_tokenizer(f"<WM_{wm_idx}>", return_tensors="pt").input_ids[0, 0]
                    tidx = (batch_input_ids[bidx] == wm_token_id).nonzero(as_tuple=True)[0]

                    wm_observation = wm_observations[wm_idx].reshape(-1)
                    wm_embedding = self.observation_autoencoder.encode(wm_observation)
                    batch_input_embeds[bidx][tidx] = wm_embedding
                    
                    wm_token_bidx.append(bidx)
                    wm_token_tidx.append(tidx)
                    wm_target_embeds.append(wm_embedding)

                    # we should also mask the WM tokens for classification loss
                    batch_obs_mask[bidx, tidx] = True

        batch_obs_data['wm_token_bidx'] = wm_token_bidx
        batch_obs_data['wm_token_tidx'] = wm_token_tidx
        if len(wm_target_embeds) > 0:
            batch_obs_data['wm_target_embeds'] = torch.stack(wm_target_embeds)
        else:
            batch_obs_data['wm_target_embeds'] = None
        return batch_input_embeds, batch_obs_data

    def obtain_ignore_mask(self, batch_input_ids: torch.Tensor, batch_attention_mask: torch.Tensor, batch_obs_mask: torch.Tensor, batch_index_data: dict):
        ignore_mask = super().obtain_ignore_mask(batch_input_ids, batch_attention_mask, batch_obs_mask, batch_index_data)

        if self.mask_collision_action:
            collision_action_index = batch_index_data['collision_action_index']
            for bidx, atidxs in enumerate(collision_action_index):
                for tidx in atidxs:
                    ignore_mask[bidx, tidx] = True

        return ignore_mask

    def obtain_task_masks(self, batch_input_ids: torch.Tensor, batch_ignore_mask: torch.Tensor):
        task_masks = super().obtain_task_masks(batch_input_ids, batch_ignore_mask)

        backspace_token_id = self.llm_tokenizer("<BACKSPACE>", return_tensors="pt").input_ids[0, 0]
        commit_token_id = self.llm_tokenizer("<COMMIT>", return_tensors="pt").input_ids[0, 0]

        separator_tokens = [backspace_token_id, commit_token_id]
        
        if self.use_wm:
            bwm_token_id = self.llm_tokenizer("<BWM>", return_tensors="pt").input_ids[0, 0]
            ewm_token_id = self.llm_tokenizer("<EWM>", return_tensors="pt").input_ids[0, 0]
            separator_tokens += [bwm_token_id, ewm_token_id]

        for token_id in separator_tokens:
            task_masks['separator'] |= batch_input_ids == token_id

        task_masks['cot'] = ~task_masks['separator'] & ~task_masks['rollout_stop'] & ~task_masks['action'] & ~batch_ignore_mask
        task_masks['cot'] = task_masks['cot'].bool()

        return task_masks
    
    def compute_loss(self, llm_output: BaseModelOutputWithPast, batch_label_ids: torch.Tensor, task_masks: dict[str, torch.Tensor], obs_flatten: torch.Tensor, obs_embed: torch.Tensor, valid_mask: torch.Tensor, batch_obs_data: dict) -> dict[str, torch.Tensor]:
        loss_dict = super().compute_loss(llm_output, batch_label_ids, task_masks, obs_flatten, obs_embed, valid_mask, batch_obs_data)

        if self.to_train_wm and batch_obs_data['wm_target_embeds'] is not None:
            pred_hidden_states = llm_output.hidden_states[-1]
            
            wm_pred_bidx = batch_obs_data['wm_token_bidx']
            wm_pred_tidx = [tidx - 1 for tidx in batch_obs_data['wm_token_tidx']] # shift target by 1 step

            wm_pred_hidden_states = pred_hidden_states[wm_pred_bidx, wm_pred_tidx]
            
            wm_pred_embeds = self.wm_head(wm_pred_hidden_states)
            wm_target_embeds = batch_obs_data['wm_target_embeds']

            wm_loss = F.mse_loss(wm_pred_embeds, wm_target_embeds, reduction="mean")
            loss_dict['wm'] = wm_loss

            loss_dict['total'] += wm_loss * self.loss_weight['wm']

        return loss_dict

    def init_action_inference(self, past_input_embeds: torch.Tensor | None, past_input_str: str, curr_obs: torch.Tensor, generate_cfg: dict):
        '''
        Conduct the initial action inference step given the current observation, end with <EOA> or <EndOfRollout>.
        '''

        curr_obs_idx = past_input_str.count("<Obs_")
        curr_obs_str = f"<BOO><Obs_{curr_obs_idx}><EOO>"
        
        curr_input_str = curr_obs_str # do not enforce <BOA> and allow <EndOfRollout>
        curr_input_embeds = self.llm_backbone.get_input_embeddings()(self.llm_tokenizer(curr_input_str, return_tensors="pt").input_ids.to(curr_obs.device))
        
        curr_obs_embed = self.observation_autoencoder.encode(curr_obs.reshape(1, -1)) # 1, hidden_dim
        curr_input_embeds[0, 1] = curr_obs_embed

        input_embeds = torch.cat([past_input_embeds, curr_input_embeds], dim=1)

        eoa_token_id = self.llm_tokenizer("<EOA>", return_tensors="pt").input_ids[0, 0]
        curr_output = self.llm_backbone.generate(inputs_embeds=input_embeds, use_cache=False, past_key_values=None, return_dict_in_generate=True, eos_token_id=eoa_token_id, **generate_cfg)

        new_generated_ids = curr_output.sequences[0]
        new_generated_str = self.llm_tokenizer.decode(new_generated_ids, skip_special_tokens=False)
        new_generated_embeds = self.llm_backbone.get_input_embeddings()(new_generated_ids)

        update_str = curr_input_str + new_generated_str
        update_embeddings = torch.cat([curr_input_embeds, new_generated_embeds[None, :]], dim=1)

        return update_str, update_embeddings

    def cot_start_inference(self, past_input_embeds: torch.Tensor | None, past_input_str: str, cot_mode: str, use_wm: bool):
        '''
        Decide to do CoT or not, different modes: 
            Use WM:
                    - Pred: let model predict one token, check <BWM> and <COMMIT> logit; select the one with higher logit
                    - Always CoT: insert <BWM>
                    - Never CoT: insert <COMMIT>
            Not use WM:
                    - Pred: do nothing, pass.
                    - Always CoT: insert <BOT>
                    - Never CoT: insert <COMMIT>
        '''
        if not use_wm:
            if cot_mode == 'pred':
                return "", None
            elif cot_mode == 'always':
                curr_input_str = "<BOT>"
            elif cot_mode == 'never':
                curr_input_str = "<COMMIT>"
        else:
            if cot_mode == 'pred':
                llm_output = self.llm_backbone.forward(inputs_embeds=past_input_embeds)
                llm_logits = llm_output.logits[0, -1, :]

                bwm_token_id = self.llm_tokenizer("<BWM>", return_tensors="pt").input_ids[0, 0].item()
                commit_token_id = self.llm_tokenizer("<COMMIT>", return_tensors="pt").input_ids[0, 0].item()

                bwm_logit = llm_logits[bwm_token_id]
                commit_logit = llm_logits[commit_token_id]

                if bwm_logit > commit_logit:
                    curr_input_str = "<BWM>"
                else:
                    curr_input_str = "<COMMIT>"
            elif cot_mode == 'always':
                curr_input_str = "<BWM>"
            elif cot_mode == 'never':
                curr_input_str = "<COMMIT>"

        # insert the CoT token
        curr_input_embeds = self.llm_backbone.get_input_embeddings()(self.llm_tokenizer(curr_input_str, return_tensors="pt").input_ids.to(past_input_embeds.device))
        update_str = curr_input_str
        update_embeddings = curr_input_embeds

        return update_str, update_embeddings

    def cot_append_wm_embeddings(self, past_input_embeds: torch.Tensor, past_input_str: str, wm_obs: torch.Tensor|None):
        '''
        Append the WM embeddings to the past input embeddings, the last token is <BWM>.

        If use simulator WM, encode the WM observation.
        If use model WM, do WM prediction with the last hidden state.
        '''
        
        if wm_obs is not None:
            wm_embed = self.observation_autoencoder.encode(wm_obs.reshape(1, -1))
        else:
            llm_output = self.llm_backbone.forward(inputs_embeds=past_input_embeds, output_hidden_states=True)
            wm_pred_input = llm_output.hidden_states[-1][:, -1, :]
            wm_embed = self.wm_head(wm_pred_input)

        ewm_token_id = self.llm_tokenizer("<EWM>", return_tensors="pt").input_ids[0, 0].item()
        ewm_token_embed = self.llm_backbone.get_input_embeddings()(torch.tensor([ewm_token_id], device=past_input_embeds.device))

        wm_cnt = past_input_str.count("<WM_")
        update_str = f"<WM_{wm_cnt}>" + "<EWM>"
        update_embeddings = torch.cat([wm_embed[None, :], ewm_token_embed[None, :]], dim=1)

        return update_str, update_embeddings

    def cot_commit_inference(self, past_input_embeds: torch.Tensor, past_input_str: str, generate_cfg: dict):
        commit_token_id = self.llm_tokenizer("<COMMIT>", return_tensors="pt").input_ids[0, 0]

        curr_output = self.llm_backbone.generate(inputs_embeds=past_input_embeds, use_cache=False, past_key_values=None, return_dict_in_generate=True, eos_token_id=commit_token_id, **generate_cfg)

        new_generated_ids = curr_output.sequences[0]
        new_generated_str = self.llm_tokenizer.decode(new_generated_ids, skip_special_tokens=False)
        new_generated_embeds = self.llm_backbone.get_input_embeddings()(new_generated_ids)

        update_str = new_generated_str
        update_embeddings = new_generated_embeds[None, :]

        return update_str, update_embeddings