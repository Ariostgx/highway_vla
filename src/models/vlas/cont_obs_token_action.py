from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from .base import BaseVLA
from ..backbones.observation import VectorObservationAutoencoder
from ..backbones.mlp import MLP

class ContObsTokenActionVLA(BaseVLA):
    def __init__(self, llm_backbone: PreTrainedModel, obs_dim: int, num_actions: int, hidden_dim: int, mlp_layers: int, loss_weight: dict[str, float] = {"action": 1.0, "obs": 1.0, 'reconst': 1.0}):
        super().__init__(llm_backbone)
        self.observation_autoencoder = VectorObservationAutoencoder(obs_dim, hidden_dim, mlp_layers)
        self.action_embed = nn.Embedding(num_actions, hidden_dim)
        self.action_pred = MLP(hidden_dim, hidden_dim, num_actions, mlp_layers, output_activation=None)
        self.loss_weight = loss_weight
        self.num_actions = num_actions

        # define special tokens
        self.boo_token_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01) # [BOO]
        self.eoo_token_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01) # [EOO]

        self.to_predict_action = loss_weight["action"] > 0
        self.to_predict_obs = loss_weight["obs"] > 0
        self.to_reconstruct_obs = loss_weight["reconst"] > 0

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, valid_mask: torch.Tensor, inference: bool = False) -> Tuple[BaseModelOutputWithPast, torch.Tensor]:
        B, T = observations.shape[:2]
        
        # flatten the observations and actions
        observations = observations.reshape(B, T, -1)
        actions = actions.reshape(B, T).long()
        valid_mask = valid_mask.reshape(B, T).bool()

        # autoencode the observations
        obs_embed = self.observation_autoencoder.encode(observations) # B, T, hidden_dim
        
        # embed the actions
        actions[~valid_mask] = 0
        act_embed = self.action_embed(actions) # B, T, hidden_dim

        # process the input 
        all_embed, all_mask = self.process_emd(obs_embed, act_embed, valid_mask) # B, 4T, hidden_dim

        # remove the last token (Action_token_T-1)
        # [BOO] Obs_token_0 [EOO] Act_token_0 [BOO] Obs_token_1 [EOO] Act_token_1 ... [EOO]
        input_embed = all_embed[:, :-1] # B, 4T-1, hidden_dim
        input_mask = all_mask[:, :-1] # B, 4T-1

        # forward pass
        outputs = self.llm_backbone(inputs_embeds=input_embed, attention_mask=input_mask, output_hidden_states=True)

        # obtain predicted features for the next tokens
        # Obs_token_0 [EOO] Act_token_0 [BOO] Obs_token_1 [EOO] Act_token_1 ... [EOO] Action_token_T-1
        output_embed = outputs.hidden_states[-1] # B, 4T-1, hidden_dim

        predictions = self._obtain_predictions(input_embed, output_embed, inference)

        loss_dict = self.compute_loss(all_embed, predictions, observations, actions, valid_mask, inference)

        return outputs, predictions, loss_dict

    def _obtain_predictions(self, input_embed: torch.Tensor, output_embed: torch.Tensor, inference: bool = False) -> torch.Tensor:
        predictions = {}
        # input_embed
        # [BOO] Obs_token_0 [EOO] Act_token_0 [BOO] Obs_token_1 [EOO] Act_token_1 ... Obs_token_T-1 [EOO]
        # output_embed
        # Obs_token_0 [EOO] Act_token_0 [BOO] Obs_token_1 [EOO] Act_token_1 ... [EOO] Action_token_T-1

        if self.to_predict_action:
            predictions["action"] = self.action_pred(output_embed[:, 2::4]) # B, T, num_actions

        if self.to_predict_obs and not inference:
            predictions["obs"] = self.observation_autoencoder.decode(output_embed[:, ::4]) # B, T, obs_dim
        
        if self.to_reconstruct_obs and not inference:
            predictions["reconst"] = self.observation_autoencoder.decode(input_embed[:, 1::4]) # B, T, obs_dim

        return predictions

    def compute_loss(self, all_embed: torch.Tensor, predictions: dict[str, torch.Tensor], observations: torch.Tensor, actions: torch.Tensor, valid_mask: torch.Tensor, inference: bool = False) -> dict[str, torch.Tensor]:
        if inference:
            return {}

        total_loss = 0
        loss_dict = {}

        if self.to_predict_action:
            num_class = predictions["action"].shape[-1]
            action_loss = F.cross_entropy(predictions["action"].reshape(-1, num_class), actions.reshape(-1), reduction="none") # B, T
            action_mask = valid_mask.reshape(-1)
            action_loss = (action_loss * action_mask).sum() / action_mask.sum()
            total_loss += action_loss * self.loss_weight["action"]
            loss_dict["action"] = action_loss

        if self.to_predict_obs:
            # ignore the first observation token as it cannot be predicted without any context
            obs_loss = F.mse_loss(predictions["obs"][:, 1:], observations[:, 1:], reduction="none").mean(dim=-1) # B, T - 1
            obs_loss = (obs_loss * valid_mask[:, 1:]).sum() / valid_mask[:, 1:].sum()
            total_loss += obs_loss * self.loss_weight["obs"]
            loss_dict["obs"] = obs_loss

        if self.to_reconstruct_obs:
            reconst_loss = F.mse_loss(predictions["reconst"], observations, reduction="none").mean(dim=-1) # B, T
            reconst_loss = (reconst_loss * valid_mask).sum() / valid_mask.sum()
            total_loss += reconst_loss * self.loss_weight["reconst"]
            loss_dict["reconst"] = reconst_loss
        
        loss_dict["total"] = total_loss

        return loss_dict

    def process_emd(self, obs_embed: torch.Tensor, act_embed: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # interleave the obs and act
        # [BOO] Obs_token_0 [EOO] Act_token_0 [BOO] Obs_token_1 [EOO] Act_token_1 ...

        B, T, D = obs_embed.shape
        x = torch.zeros(B, 4*T, D, device=obs_embed.device)
        x[:, ::4] = self.boo_token_embed
        x[:, 1::4] = obs_embed
        x[:, 2::4] = self.eoo_token_embed
        x[:, 3::4] = act_embed

        mask = torch.zeros(B, 4*T, device=valid_mask.device, dtype=torch.bool)
        mask[:, ::4] = True
        mask[:, 1::4] = valid_mask
        mask[:, 2::4] = True
        mask[:, 3::4] = valid_mask

        return x, mask

    def predict_action(self, observations: torch.Tensor, past_actions: torch.Tensor) -> torch.Tensor:
        '''
        predict the next action given the past actions and the current observation during inference/rollout.

        args:
            observations: T, ...
            past_actions: T-1
        returns:
            predictions: 1, num_actions
        '''

        with torch.no_grad():
            T = observations.shape[0]
            
            obs_input = observations[None, ...] # 1, T, ...

            # pad an empty action token at the end to align with the input format
            act_input = past_actions[None, ...] # 1, T-1
            act_input = torch.cat([act_input, torch.zeros(1, 1, device=act_input.device)], dim=1) # 1, T

            valid_mask = torch.ones(1, T, device=past_actions.device, dtype=torch.bool) # 1, T

            _, predictions, _ = self.forward(obs_input, act_input, valid_mask, inference=True)

            return F.softmax(predictions['action'][0, -1], dim=-1)
