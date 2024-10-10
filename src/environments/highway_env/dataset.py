# build a dataset using cached observations and actions
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class HighwayDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = self._obtain_all_files()

    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        observations = torch.tensor(data['observations'], dtype=torch.float32)
        actions = torch.tensor(data['actions'], dtype=torch.float32)
        return observations, actions
    
    def _obtain_all_files(self):
        return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.npz')]

# define the collate function
def collate_fn(batch):
  # Pad sequences to the maximum length in the batch
  max_obs_len = max(obs.shape[0] for obs, _ in batch)
  max_act_len = max(act.shape[0] for _, act in batch)

  # Pad and stack observations
  observations = torch.stack([
      torch.nn.functional.pad(obs, (0, 0, 0, 0, 0, max_obs_len - obs.shape[0]), value=-100)
      for obs, _ in batch
  ])

  # Pad and stack actions
  actions = torch.stack([
      torch.nn.functional.pad(act, (0, max_act_len - act.shape[0]), value=-100)
      for _, act in batch
  ])


  valid_mask = torch.ones(len(batch), max_obs_len, dtype=torch.bool)
  for i, (obs, _) in enumerate(batch):
    valid_mask[i, obs.shape[0]:] = False # mark the invalid tokens

  return observations, actions, valid_mask


if __name__ == '__main__':

  data_folder = '/storage/Datasets/highway_env/highway_fast_v0_dqn_meta_action/rollouts'

  dataset = HighwayDataset(data_folder)

  # define the dataloader
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
