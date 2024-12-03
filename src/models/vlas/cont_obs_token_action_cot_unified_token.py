from locale import currency
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache

from .base import BaseVLA
from ..backbones.mlp import MLP
from ..backbones.observation import VectorObservationAutoencoder

from ...auto_labeling.highway_env.lane_change import LaneChangeTaskSpec

class ContObsTokenActionCOTVLAUnifiedToken(BaseVLA):
    def __init__(self, llm_backbone: PreTrainedModel, llm_tokenizer: PreTrainedTokenizer, task_spec_func: LaneChangeTaskSpec, obs_dim: int, num_actions: int, hidden_dim: int, mlp_layers: int, loss_weight: dict[str, float] = {"action": 1.0, 'reconst': 1.0, "cot": 1.0, "separator": 1.0, "rollout_stop": 1.0}, cot_mode: str = "none", cot_cfg: dict = {}, max_obs_len: int = 50):
        super().__init__(llm_backbone)

        assert cot_mode in ["none", "start", "all"]
        
        self.loss_weight = loss_weight
        self.num_actions = num_actions
        self.max_obs_len = max_obs_len
        self.llm_tokenizer = llm_tokenizer
        self.task_spec_func = task_spec_func
        self.cot_mode = cot_mode
        self.cot_cfg = cot_cfg


        # set up the continuous observation autoencoder
        self.observation_autoencoder = VectorObservationAutoencoder(obs_dim, hidden_dim, mlp_layers)
        
        # define special separator tokens and action tokens
        special_modal_tokens = ['<BOO>', '<EOO>', '<BOT>', '<EOT>', '<BOA>', '<EOA>', '<EndOfRollout>']
        special_action_tokens = [f'<Act_{i}>' for i in range(num_actions)]
        obs_placeholder_tokens = [f'<Obs_{i}>' for i in range(max_obs_len)]
        special_tokens = special_modal_tokens + special_action_tokens + obs_placeholder_tokens

        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.llm_tokenizer.pad_token = '<PAD>'
        self.llm_backbone.resize_token_embeddings(len(self.llm_tokenizer))

        self.to_reconstruct_obs = loss_weight["reconst"] > 0

    def obtain_input_strs(self, batch_data: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        '''
        Obtain the input strings for the LLM.
        Afterwards, needs to:
            1. Replace the observation placeholder tokens with actual observation embeddings.
            2. Mask labels for non-training tokens: <BOO>, <EOO>, GoalText tokens
        Output:
            batch_input_strs: list[str], the input strings for the LLM
            batch_index_data: dict, the index of the goal specification in the input strings
        '''
        observations, actions, valid_mask = batch_data[:3]
        B = observations.shape[0]
        batch_input_strs = []
        batch_goal_spec_index = []

        for bidx in range(B):
            # the input string for the LLM. need to replace the observation placeholder tokens with actual observation strings after encoding
            input_str = "" 
            task_spec = self.task_spec_func(observations[bidx][valid_mask[bidx]].cpu().numpy(), actions[bidx][valid_mask[bidx]].cpu().numpy(), self.cot_cfg)
            
            goal_spec = task_spec.get_goal_spec()
            cot_prompt = task_spec.get_multi_step_cot_prompt()
            hop_indices = task_spec.get_task_hop_info()['hop_indices']

            last_hop_idx = hop_indices[-1]

            input_str += goal_spec
            goal_spec_token_num = self.llm_tokenizer(goal_spec, return_tensors="pt").input_ids.shape[1]
            batch_goal_spec_index.append(goal_spec_token_num)

            for obs_idx in range(last_hop_idx + 1):
                # only add observation + action at the last hop
                if not valid_mask[bidx, obs_idx]:
                    continue
                
                # step 1: always add observation
                input_str += f"<BOO><Obs_{obs_idx}><EOO>"

                # step 2: decide whether to do CoT or Action after observation <BOO> token
                ignore_cot = (self.cot_mode == "none") or (self.cot_mode == "start" and obs_idx > 0)
                missing_cot = obs_idx not in cot_prompt.keys()
                
                if not (ignore_cot or missing_cot):
                    # add CoT after observation <BOO> token
                    cot_text = "<BOT>" + cot_prompt[obs_idx] + "<EOT>"
                    input_str += cot_text
                
                # step 3: add action tokens
                act_id = actions[bidx, obs_idx].cpu().long().item()
                input_str += f"<BOA><Act_{act_id}><EOA>"

            input_str += f"<BOO><Obs_{last_hop_idx + 1}><EOO><EndOfRollout>"
            batch_input_strs.append(input_str)

        batch_index_data = {'goal_spec_index': batch_goal_spec_index}

        return batch_input_strs, batch_index_data

    def replace_obs_placeholder_tokens(self, batch_input_ids: torch.Tensor, batch_input_embeds: torch.Tensor, obs_embed: torch.Tensor, batch_index_data: dict):
        B, T = batch_input_ids.shape[:2]

        batch_obs_mask = torch.zeros(B, T, device=batch_input_ids.device, dtype=torch.bool)

        for bidx in range(B):
            for obs_idx in range(self.max_obs_len):
                obs_token_id = self.llm_tokenizer(f"<Obs_{obs_idx}>", return_tensors="pt").input_ids[0, 0]
                obs_token_mask = batch_input_ids[bidx] == obs_token_id
                if obs_token_mask.any():
                    batch_input_embeds[bidx][obs_token_mask] = obs_embed[bidx, obs_idx]
                    batch_obs_mask[bidx][obs_token_mask] = True
        
        batch_obs_data = {'obs_mask': batch_obs_mask}
        
        return batch_input_embeds, batch_obs_data
    
    def obtain_ignore_mask(self, batch_input_ids: torch.Tensor, batch_attention_mask: torch.Tensor, batch_obs_mask: torch.Tensor, batch_index_data: dict):
        # ignore goal specification text tokens during training
        goal_spec_token_mask = torch.zeros_like(batch_attention_mask, dtype=torch.bool)
        for bidx, goal_spec_idx in enumerate(batch_index_data['goal_spec_index']):
            goal_spec_token_mask[bidx, :goal_spec_idx] = True

        # ignore the <BOO> and <EOO> tokens during training
        boo_token_id = self.llm_tokenizer("<BOO>", return_tensors="pt").input_ids[0, 0]
        eoo_token_id = self.llm_tokenizer("<EOO>", return_tensors="pt").input_ids[0, 0]
        separator_token_mask = (batch_input_ids == boo_token_id) | (batch_input_ids == eoo_token_id)

        ignore_mask = goal_spec_token_mask | separator_token_mask | batch_obs_mask | ~(batch_attention_mask.bool())

        return ignore_mask

    def obtain_task_masks(self, batch_input_ids: torch.Tensor, batch_ignore_mask: torch.Tensor):
        # obtain the masks for the task-specific tokens
        # 1. separator: <BOT>, <EOT>, <BOA> and <EOA> tokens for separator tokens
        # 2. end_of_rollout: <EndOfRollout> token for rollout stop
        # 3. action: <Act_i> tokens for action
        # 4. cot: the rest of the tokens for CoT text

        task_masks = {}

        B, T = batch_input_ids.shape[:2]

        bot_token_id = self.llm_tokenizer("<BOT>", return_tensors="pt").input_ids[0, 0]
        eot_token_id = self.llm_tokenizer("<EOT>", return_tensors="pt").input_ids[0, 0]
        boa_token_id = self.llm_tokenizer("<BOA>", return_tensors="pt").input_ids[0, 0]
        eoa_token_id = self.llm_tokenizer("<EOA>", return_tensors="pt").input_ids[0, 0]
        end_of_rollout_token_id = self.llm_tokenizer("<EndOfRollout>", return_tensors="pt").input_ids[0, 0]

        task_masks['separator'] = torch.zeros(B, T, device=batch_input_ids.device, dtype=torch.bool)
        for token_id in [bot_token_id, eot_token_id, boa_token_id, eoa_token_id]:
            task_masks['separator'] |= batch_input_ids == token_id

        task_masks['rollout_stop'] = batch_input_ids == end_of_rollout_token_id

        task_masks['action'] = torch.zeros(B, T, device=batch_input_ids.device, dtype=torch.bool)
        for act_id in range(self.num_actions):
            act_token_id = self.llm_tokenizer(f"<Act_{act_id}>", return_tensors="pt").input_ids[0, 0]
            task_masks['action'] |= batch_input_ids == act_token_id
        
        task_masks['cot'] = ~task_masks['separator'] & ~task_masks['rollout_stop'] & ~task_masks['action'] & ~batch_ignore_mask
        task_masks['cot'] = task_masks['cot'].bool()
        
        return task_masks


    def forward(self, batch_data: Tuple[torch.Tensor]) -> Tuple[BaseModelOutputWithPast, torch.Tensor]:
        observations, actions, valid_mask = batch_data[:3]

        B, T = observations.shape[:2]
        
        # flatten the observations and actions
        obs_flatten = observations.reshape(B, T, -1)
        act_flatten = actions.reshape(B, T).long()
        valid_mask = valid_mask.reshape(B, T).bool()

        # embed the actions
        act_flatten[~valid_mask] = 0

        # process the input strings
        batch_input_strs, batch_index_data = self.obtain_input_strs(batch_data)

        # obtain batch_input_ids and batch_attention_mask
        tokenized = self.llm_tokenizer(batch_input_strs, return_tensors="pt", padding=True, truncation=True).to(observations.device)
        batch_input_ids = tokenized.input_ids
        batch_attention_mask = tokenized.attention_mask

        # tokens up to t-1 for input forward pass
        batch_input_embeds = self.llm_backbone.get_input_embeddings()(batch_input_ids)
        batch_attention_mask = batch_attention_mask

        # autoencode the observations
        obs_embed = self.observation_autoencoder.encode(obs_flatten) # B, T, hidden_dim
        batch_input_embeds, batch_obs_data = self.replace_obs_placeholder_tokens(batch_input_ids, batch_input_embeds, obs_embed, batch_index_data)
        
        batch_label_ignore_mask = self.obtain_ignore_mask(batch_input_ids, batch_attention_mask, batch_obs_data['obs_mask'], batch_index_data)
        batch_label_ids = batch_input_ids.clone()
        batch_label_ids[batch_label_ignore_mask] = -100

        task_masks = self.obtain_task_masks(batch_input_ids, batch_label_ignore_mask)

        batch_input_embeds = batch_input_embeds[:, :-1]
        batch_attention_mask = batch_attention_mask[:, :-1]
        batch_label_ids = batch_label_ids[:, 1:]

        llm_output = self.llm_backbone(inputs_embeds=batch_input_embeds, attention_mask=batch_attention_mask, output_hidden_states=True)

        loss_dict = self.compute_loss(llm_output, batch_label_ids, task_masks, obs_flatten, obs_embed, valid_mask, batch_obs_data)
        
        # return loss_dict, batch_input_embeds
        return loss_dict, batch_input_embeds, batch_label_ids, batch_input_ids, llm_output

    def compute_loss(self, llm_output: BaseModelOutputWithPast, batch_label_ids: torch.Tensor, task_masks: dict[str, torch.Tensor], obs_flatten: torch.Tensor, obs_embed: torch.Tensor, valid_mask: torch.Tensor, batch_obs_data: dict) -> dict[str, torch.Tensor]:
        total_loss = 0
        loss_dict = {}

        device = llm_output.logits.device

        B = batch_label_ids.shape[0]
        all_token_loss = F.cross_entropy(llm_output.logits.reshape(-1, self.llm_backbone.config.vocab_size), batch_label_ids.reshape(-1).to(device), reduction="none")
        all_token_loss = all_token_loss.reshape(B, -1)

        for task_name, task_mask in task_masks.items():
            if not task_mask.any():
                loss_dict[task_name] = 0
                continue
            
            task_loss = all_token_loss[task_mask[:, 1:]].mean()
            total_loss += task_loss * self.loss_weight[task_name]
            loss_dict[task_name] = task_loss

        if self.to_reconstruct_obs:
            obs_reconst = self.observation_autoencoder.decode(obs_embed)
            reconst_loss = F.mse_loss(obs_reconst, obs_flatten, reduction="none").mean(dim=-1)
            loss_dict["reconst"] = (reconst_loss * valid_mask).sum() / valid_mask.sum()

            total_loss += loss_dict["reconst"] * self.loss_weight["reconst"]

        loss_dict["total"] = total_loss

        return loss_dict

    def inference_step(self, past_input_embeds: torch.Tensor | None, past_input_str: str, past_key_values: Cache, curr_obs: torch.Tensor, cot_infernce_mode: str, generate_cfg: dict):
        '''
        Conduct a single inference step given past sequence and the current observation.

        Inputs:
            past_input_embeds: the input embeddings for the LLM before the current step
            past_input_str: the input string for the LLM before the current step
            past_key_value: the key/value cache for the LLM
            curr_obs: the current observation
            cot_mode: the CoT mode
            generate_cfg: the generation configuration
        '''

        curr_obs_idx = past_input_str.count("<Obs_")
        curr_obs_str = f"<BOO><Obs_{curr_obs_idx}><EOO>"
        
        use_cot = (cot_infernce_mode == 'always') or (cot_infernce_mode == 'start' and curr_obs_idx == 0)

        if cot_infernce_mode == 'never':
            curr_input_str = curr_obs_str + "<BOA>"
        elif cot_infernce_mode == 'pred' or not use_cot:
            curr_input_str = curr_obs_str
        else:
            curr_input_str = curr_obs_str + "<BOT>"
        
        # replace the observation placeholder token with the actual observation embedding
        curr_input_embeds = self.llm_backbone.get_input_embeddings()(self.llm_tokenizer(curr_input_str, return_tensors="pt").input_ids.to(curr_obs.device))
        curr_obs_embed = self.observation_autoencoder.encode(curr_obs.reshape(1, -1)) # 1, hidden_dim
        curr_input_embeds[0, 1] = curr_obs_embed

        if past_input_embeds is None:
            input_embeds = curr_input_embeds
        else:
            input_embeds = torch.cat([past_input_embeds, curr_input_embeds], dim=1)

        # generate the sequence up to the <EOA> token
        eoa_token_id = self.llm_tokenizer("<EOA>", return_tensors="pt").input_ids[0, 0]
        # curr_output = self.llm_backbone.generate(inputs_embeds=input_embeds, use_cache=True, past_key_values=past_key_values, return_dict_in_generate=True, eos_token_id=eoa_token_id, **generate_cfg)
        curr_output = self.llm_backbone.generate(inputs_embeds=input_embeds, use_cache=False, past_key_values=None, return_dict_in_generate=True, eos_token_id=eoa_token_id, **generate_cfg)

        new_generated_ids = curr_output.sequences[0]
        new_generated_str = self.llm_tokenizer.decode(new_generated_ids, skip_special_tokens=False)
        new_generated_embeds = self.llm_backbone.get_input_embeddings()(new_generated_ids)

        update_str = curr_input_str + new_generated_str
        update_embeddings = torch.cat([curr_input_embeds, new_generated_embeds[None, :]], dim=1)

        return update_str, update_embeddings