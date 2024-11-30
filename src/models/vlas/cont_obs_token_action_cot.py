from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..backbones.mlp import MLP
from .cont_obs_token_action import ContObsTokenActionVLA
from ...auto_labeling.highway_env.lane_change import LaneChangeTaskSpec

class ContObsTokenActionCOTVLA(ContObsTokenActionVLA):
    def __init__(self, llm_backbone: PreTrainedModel, llm_tokenizer: PreTrainedTokenizer, task_spec_func: LaneChangeTaskSpec, obs_dim: int, num_actions: int, hidden_dim: int, mlp_layers: int, loss_weight: dict[str, float] = {"action": 1.0, "obs": 1.0, 'reconst': 1.0, "cot": 1.0, "cot_cls": 1.0, "rollout_stop": 1.0}, cot_mode: str = "none", cot_cfg: dict = {}):
        super().__init__(llm_backbone, obs_dim, num_actions, hidden_dim, mlp_layers, loss_weight)
        self.llm_tokenizer = llm_tokenizer

        # define special tokens
        # [BOO] Obs [EOO] [BOA] Act [EOA] [BOO]
        self.boa_token_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01) # [BOA]
        self.eoa_token_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01) # [EOA]
        self.bot_token_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01) # [BOT]
        
        # binary classification head to predict whether do CoT or Action after observation [BOO] token
        # binary classification head to predict whether to stop rollout
        self.cot_start_head = MLP(hidden_dim, hidden_dim, 2, 1, output_activation=None)
        self.rollout_stop_head = MLP(hidden_dim, hidden_dim, 2, 1, output_activation=None)
        
        # add special EOT token to the tokenizer because it will be predicted among with text tokens
        special_tokens = {
            'additional_special_tokens': ['[EOT]']
        }
        self.llm_tokenizer.add_special_tokens(special_tokens)
        self.llm_backbone.resize_token_embeddings(len(self.llm_tokenizer))
        
        self.task_spec_func = task_spec_func
        self.cot_mode = cot_mode
        self.cot_cfg = cot_cfg

        assert self.cot_mode in ["none", "start", "all"]

        self.to_predict_cot = loss_weight["cot"] > 0 and self.cot_mode != "none"
        self.to_predict_rollout_stop = loss_weight["rollout_stop"] > 0

    def process_emd(self, obs_embed: torch.Tensor, act_embed: torch.Tensor, valid_mask: torch.Tensor, observations: torch.Tensor, actions: torch.Tensor):
    # -> Tuple[torch.Tensor, torch.Tensor, dict]:
        '''
        Process the embeddings of observations, actions, and valid mask.
        Interleave the obs, cot, act tokens for each batch item.

        Output:
            x: torch.Tensor, shape (B, T_MAX, D)
            mask: torch.Tensor, shape (B, T_MAX)
            index_labels: dict, the index and ground-truth labels for each type of token
        '''
        B, T, D = obs_embed.shape
        batch_seq_emds = []
        batch_index_labels = []
        batch_preview_strs = []

        for bidx in range(B):
            seq_emds = [] # a list of embeddings for the sequence, shapes are all in [t, D]. 
            seq_last_idx = 0
            seq_preview_str = ""
            
            label_types = ['act', 'cot', 'cot_cls', 'rollout_stop']
            index_labels = {type_name: {'index': [], 'label': []} for type_name in label_types}

            task_spec = self.task_spec_func(observations[bidx].cpu().numpy(), actions[bidx].cpu().numpy(), self.cot_cfg)
            
            goal_spec = task_spec.get_goal_spec()
            cot_prompt = task_spec.get_multi_step_cot_prompt()
            hop_indices = task_spec.get_task_hop_info()['hop_indices']

            # print("cot_prompt", cot_prompt.keys())

            last_hop_idx = hop_indices[-1]

            _, goal_embed = self._obtain_text_ids_embedding(goal_spec)
            seq_emds.append(goal_embed)
            seq_last_idx += goal_embed.shape[0]
            seq_preview_str += goal_spec + " "
            
            for obs_idx in range(last_hop_idx + 2):
            # only add observation + action at the last hop

                if not valid_mask[bidx, obs_idx]:
                    continue
                # step 1: always add observation
                obs_emd = obs_embed[bidx, obs_idx]
                full_obs_emd = torch.stack([self.boo_token_embed[0, 0], obs_emd, self.eoo_token_embed[0, 0]], dim=0) # [3, D]
                
                seq_emds.append(full_obs_emd)
                seq_last_idx += full_obs_emd.shape[0]
                seq_preview_str += f"[BOO] <Obs_{obs_idx}> [EOO] "

                # stop rollout after the last hop and get the goal as observation
                index_labels['rollout_stop']['index'].append(seq_last_idx - 1)
                if obs_idx == last_hop_idx + 1:
                    index_labels['rollout_stop']['label'].append(torch.tensor(1, device=obs_emd.device))
                    break
                else:
                    index_labels['rollout_stop']['label'].append(torch.tensor(0, device=obs_emd.device))

                # step 2: decide whether to do CoT or Action after observation [BOO] token
                if self.cot_mode == "none" or obs_idx not in cot_prompt.keys():
                    use_cot = False
                elif self.cot_mode == "start" and obs_idx > 0:
                    use_cot = False
                else:
                    use_cot = True
                
                # TODO: check if this is correct
                index_labels['cot_cls']['index'].append(seq_last_idx - 1)
                index_labels['cot_cls']['label'].append(use_cot)
                
                # step 3: Add CoT after observation [BOO] token
                if use_cot:
                    # add CoT after observation [BOO] token
                    cot_text = cot_prompt[obs_idx] + "[EOT]"
                    cot_ids, cot_emd = self._obtain_text_ids_embedding(cot_text)
                    cot_emd = torch.cat([self.bot_token_embed[0], cot_emd], dim=0) # [1+N, D]
                    seq_emds.append(cot_emd)
                    
                    # ignore the first [BOT] token
                    cot_start_idx = seq_last_idx
                    cot_end_idx = cot_start_idx + cot_emd.shape[0] - 1
                    assert (cot_end_idx - cot_start_idx) == cot_ids.shape[1]

                    index_labels['cot']['index'].append(torch.arange(cot_start_idx, cot_end_idx))
                    index_labels['cot']['label'].append(cot_ids)

                    seq_last_idx += cot_emd.shape[0]
                    seq_preview_str += f"[BOT] {cot_text} "

                # step 4: add action tokens
                act_emd = act_embed[bidx, obs_idx]
                full_act_emd = torch.stack([self.boa_token_embed[0, 0], act_emd, self.eoa_token_embed[0, 0]], dim=0) # [3, D]

                index_labels['act']['index'].append(seq_last_idx)
                index_labels['act']['label'].append(actions[bidx, obs_idx])

                seq_emds.append(full_act_emd)
                seq_last_idx += full_act_emd.shape[0]
                seq_preview_str += f"[BOA] <Act_{obs_idx}> [EOA] "
        
            seq_preview_str += "[EndOfRollout]"
            batch_seq_emds.append(torch.cat(seq_emds, dim=0))
            batch_index_labels.append(index_labels)
            batch_preview_strs.append(seq_preview_str)

        batch_seq_emds_flat, batch_seq_emds_valid = self._flatten_seq_emds(batch_seq_emds)
        batch_flat_labels = self._flatten_index_labels(batch_index_labels)

        return batch_seq_emds_flat, batch_seq_emds_valid, batch_flat_labels, batch_preview_strs
            
    def _flatten_seq_emds(self, batch_seq_emds: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_seq_emds_flat = torch.nn.utils.rnn.pad_sequence(batch_seq_emds, batch_first=True)
        batch_seq_emds_valid = torch.zeros_like(batch_seq_emds_flat[..., 0], dtype=torch.bool)
        for bidx in range(len(batch_seq_emds)):
            batch_seq_emds_valid[bidx, :batch_seq_emds[bidx].shape[0]] = 1

        return batch_seq_emds_flat, batch_seq_emds_valid

    def _flatten_index_labels(self, batch_index_labels: dict) -> dict:
        batch_flat_labels = {type_name: {} for type_name in batch_index_labels[0].keys()}

        # flatten action labels
        action_bidx_list = [torch.tensor(bidx).repeat(len(batch_index_labels[bidx]['act']['index'])) for bidx in range(len(batch_index_labels))]
        batch_flat_labels['act']['bidx'] = torch.cat(action_bidx_list, dim=0)
        batch_flat_labels['act']['tidx'] = torch.cat([torch.tensor(index_labels['act']['index']) for index_labels in batch_index_labels], dim=0)
        batch_flat_labels['act']['label'] = torch.cat([torch.tensor(index_labels['act']['label']) for index_labels in batch_index_labels], dim=0)

        # flatten cot labels
        if self.cot_mode != "none":
            cot_index_flat, cot_label_flat = [], []
            cot_valid_bidx = []

            for bidx in range(len(batch_index_labels)):
                cot_index_scene = batch_index_labels[bidx]['cot']['index']
                cot_label_scene = batch_index_labels[bidx]['cot']['label']
                if len(cot_index_scene) > 0:
                    cot_index_flat.append(torch.cat(cot_index_scene, dim=0))
                    cot_label_flat.append(torch.cat(cot_label_scene, dim=1)[0])
                    cot_valid_bidx.append(bidx)

            cot_bidx_list = [torch.tensor(bidx).repeat(len(cot_index_flat[i])) for i, bidx in enumerate(cot_valid_bidx)]
            batch_flat_labels['cot']['bidx'] = torch.cat(cot_bidx_list, dim=0)
            batch_flat_labels['cot']['tidx'] = torch.cat(cot_index_flat, dim=0)
            batch_flat_labels['cot']['label'] = torch.cat(cot_label_flat, dim=0)

        # flatten cot_cls and rollout_stop labels
        for cls_type in ['cot_cls', 'rollout_stop']:
            cls_index_flat, cls_label_flat = [], []
            cls_valid_bidx = []

            for bidx in range(len(batch_index_labels)):
                cls_index_scene = batch_index_labels[bidx][cls_type]['index']
                cls_label_scene = batch_index_labels[bidx][cls_type]['label']
                if len(cls_index_scene) > 0:
                    cls_index_flat.append(torch.tensor(cls_index_scene))
                    cls_label_flat.append(torch.tensor(cls_label_scene, dtype=torch.long))
                    cls_valid_bidx.append(bidx)

            cls_bidx_list = [torch.tensor(bidx).repeat(len(cls_index_flat[i])) for i, bidx in enumerate(cls_valid_bidx)]
            batch_flat_labels[cls_type]['bidx'] = torch.cat(cls_bidx_list, dim=0)
            batch_flat_labels[cls_type]['tidx'] = torch.cat(cls_index_flat, dim=0)
            batch_flat_labels[cls_type]['label'] = torch.cat(cls_label_flat, dim=0)
        
        return batch_flat_labels
    
    def _obtain_text_ids_embedding(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.llm_backbone.device
        text_ids = self.llm_tokenizer(text, return_tensors="pt").input_ids.to(device)
        text_emd = self.llm_backbone.get_input_embeddings()(text_ids)[0]

        return text_ids, text_emd

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, valid_mask: torch.Tensor, inference: bool = False) -> Tuple[BaseModelOutputWithPast, torch.Tensor]:
        B, T = observations.shape[:2]
        
        # flatten the observations and actions
        obs_flatten = observations.reshape(B, T, -1)
        act_flatten = actions.reshape(B, T).long()
        valid_mask = valid_mask.reshape(B, T).bool()

        # autoencode the observations
        obs_embed = self.observation_autoencoder.encode(obs_flatten) # B, T, hidden_dim
        
        # embed the actions
        act_flatten[~valid_mask] = 0
        act_embed = self.action_embed(act_flatten) # B, T, hidden_dim

        # process the input 
        batch_seq_emds_flat, batch_seq_emds_valid, batch_flat_labels, batch_preview_strs = self.process_emd(obs_embed, act_embed, valid_mask, observations, act_flatten)

        # forward pass
        outputs = self.llm_backbone(inputs_embeds=batch_seq_emds_flat, attention_mask=batch_seq_emds_valid, output_hidden_states=True)

        output_logits = outputs.logits # B, T, vocab_size (used for cot text generation prediction)
        output_embed = outputs.hidden_states[-1] # B, T, hidden_dim

        predictions = self._obtain_predictions(obs_embed, output_embed, batch_flat_labels)

        loss_dict = self.compute_loss(predictions, output_logits, batch_flat_labels)

        return predictions, loss_dict, batch_preview_strs

    def _obtain_predictions(self, obs_embed: torch.Tensor, output_embed: torch.Tensor, batch_flat_labels: dict) -> dict[str, torch.Tensor]:
        predictions = {}

        if self.to_predict_action:
            bidx, tidx = batch_flat_labels['act']['bidx'], batch_flat_labels['act']['tidx']
            predictions["action"] = self.action_pred(output_embed[bidx, tidx]) # B, num_actions

        if self.to_reconstruct_obs:
            predictions["reconst"] = self.observation_autoencoder.decode(obs_embed) # B, T, obs_dim
        
        if self.to_predict_cot:
            bidx, tidx = batch_flat_labels['cot_cls']['bidx'], batch_flat_labels['cot_cls']['tidx']
            cot_cls_embed = output_embed[bidx, tidx]
            predictions["cot_cls"] = self.cot_start_head(cot_cls_embed) # B, 2
        
        if self.to_predict_rollout_stop:
            bidx, tidx = batch_flat_labels['rollout_stop']['bidx'], batch_flat_labels['rollout_stop']['tidx']
            rollout_stop_embed = output_embed[bidx, tidx]
            predictions["rollout_stop"] = self.rollout_stop_head(rollout_stop_embed) # B, 2

        return predictions

    def compute_loss(self, predictions: dict[str, torch.Tensor], output_logits: torch.Tensor, batch_flat_labels: dict) -> dict[str, torch.Tensor]:
        total_loss = 0
        loss_dict = {}

        device = predictions["action"].device

        if self.to_predict_action:
            action_label = batch_flat_labels["act"]["label"]
            action_loss = F.cross_entropy(predictions["action"].reshape(-1, self.num_actions), action_label.reshape(-1).to(device), reduction="mean")
            total_loss += action_loss * self.loss_weight["action"]
            loss_dict["action"] = action_loss
        
        if self.to_predict_rollout_stop:
            rollout_stop_label = batch_flat_labels["rollout_stop"]["label"]
            rollout_stop_loss = F.cross_entropy(predictions["rollout_stop"].reshape(-1, 2), rollout_stop_label.reshape(-1).to(device), reduction="mean")
            total_loss += rollout_stop_loss * self.loss_weight["rollout_stop"]
            loss_dict["rollout_stop"] = rollout_stop_loss
        
        if self.to_predict_cot:
            cot_cls_label = batch_flat_labels["cot_cls"]["label"]
            cot_cls_loss = F.cross_entropy(predictions["cot_cls"].reshape(-1, 2), cot_cls_label.reshape(-1).to(device), reduction="mean")
            total_loss += cot_cls_loss * self.loss_weight["cot_cls"]
            loss_dict["cot_cls"] = cot_cls_loss

            cot_text_bidx, cot_text_tidx = batch_flat_labels["cot"]["bidx"], batch_flat_labels["cot"]["tidx"]
            cot_text_logits = output_logits[cot_text_bidx, cot_text_tidx]
            cot_text_label = batch_flat_labels["cot"]["label"]
            cot_text_loss = F.cross_entropy(cot_text_logits.reshape(-1, self.llm_backbone.config.vocab_size), cot_text_label.reshape(-1).to(device), reduction="mean")
            total_loss += cot_text_loss * self.loss_weight["cot"]
            loss_dict["cot_text"] = cot_text_loss

        loss_dict["total"] = total_loss

        return loss_dict

    # def predict_action(self, observations: torch.Tensor, past_actions: torch.Tensor) -> torch.Tensor:
    #     '''
    #     predict the next action given the past actions and the current observation during inference/rollout.

    #     args:
    #         observations: T, ...
    #         past_actions: T-1
    #     returns:
    #         predictions: 1, num_actions
    #     '''

    #     with torch.no_grad():
    #         T = observations.shape[0]
            
    #         obs_input = observations[None, ...] # 1, T, ...

    #         # pad an empty action token at the end to align with the input format
    #         act_input = past_actions[None, ...] # 1, T-1
    #         act_input = torch.cat([act_input, torch.zeros(1, 1, device=act_input.device)], dim=1) # 1, T

    #         valid_mask = torch.ones(1, T, device=past_actions.device, dtype=torch.bool) # 1, T

    #         _, predictions, _ = self.forward(obs_input, act_input, valid_mask, inference=True)

    #         return F.softmax(predictions['action'][0, -1], dim=-1)
