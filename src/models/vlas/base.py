from abc import ABC
from typing import Dict

import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import PreTrainedModel

class BaseVLA(nn.Module, ABC):
    def __init__(self, llm_backbone: PreTrainedModel):
        super().__init__()
        self.llm_backbone = llm_backbone

    def predict_action(self, observations: torch.Tensor, past_actions: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
  
    def forward(self, observations: torch.Tensor, actions: torch.Tensor, valid_mask: torch.Tensor) -> BaseModelOutputWithPast:
        raise NotImplementedError