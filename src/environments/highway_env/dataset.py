# build a dataset using cached observations and actions
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class HighwayDataset(Dataset):
    def __init__(self, data_dir: str, overfit: bool = False, max_obs_len: int = 50):
        self.data_dir = data_dir
        self.overfit = overfit
        self.files = self._obtain_all_files()
        self.max_obs_len = max_obs_len

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        observations = torch.tensor(data['observations'], dtype=torch.float32)[:self.max_obs_len]
        actions = torch.tensor(data['actions'], dtype=torch.float32)[:self.max_obs_len]
        return observations, actions
    
    def _obtain_all_files(self):
        files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        files = sorted(files)

        overfit_num = 1
        
        # only use the first overfit_num files if overfit is True
        if self.overfit:
            files = files[:overfit_num] * 1600 # repeat the first 32 files to match the length
        
        return files

# define the collate function
def collate_fn(batch):
  # Pad sequences to the maximum length in the batch
  max_obs_len = max(data[0].shape[0] for data in batch)
  max_act_len = max(data[1].shape[0] for data in batch)

  # Pad and stack observations
  observations = torch.stack([
      torch.nn.functional.pad(data[0], (0, 0, 0, 0, 0, max_obs_len - data[0].shape[0]), value=-100)
      for data in batch
  ])

  # Pad and stack actions
  actions = torch.stack([
      torch.nn.functional.pad(data[1], (0, max_act_len - data[1].shape[0]), value=-100)
      for data in batch
  ])


  valid_mask = torch.ones(len(batch), max_obs_len, dtype=torch.bool)
  for i, data in enumerate(batch):
    valid_mask[i, data[0].shape[0]:] = False # mark the invalid tokens

  return observations, actions, valid_mask

class HighwayCollisionDataset(HighwayDataset):
    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        observations = torch.tensor(data['observations'], dtype=torch.float32)[:self.max_obs_len]
        actions = torch.tensor(data['actions'], dtype=torch.float32)[:self.max_obs_len]
        collision_rewind_steps = torch.tensor(data['collision_rewind_steps'], dtype=torch.int32)
        collision_observations = torch.tensor(data['collision_observations'], dtype=torch.float32)
        collision_actions = torch.tensor(data['collision_actions'], dtype=torch.float32)

        # if the collision_observations is a 1D tensor, it means no collision can be made in the sequence
        if len(collision_observations.shape) == 1:
            collision_observations = torch.zeros((0, 5, 5), dtype=torch.float32)
            collision_actions = torch.zeros((0,), dtype=torch.float32)
            collision_rewind_steps = torch.zeros((0,), dtype=torch.int32)
        
        return observations, actions, collision_rewind_steps, collision_observations, collision_actions

def collate_fn_collision(batch):
    observations, actions, valid_mask = collate_fn(batch)

    collision_rewind_steps = [data[2] for data in batch]
    collision_observations = [data[3] for data in batch]
    collision_actions = [data[4] for data in batch]

    max_collision_size = max(len(steps) for steps in collision_rewind_steps)

    collision_rewind_steps_padded = torch.stack([
        torch.nn.functional.pad(steps, (0, max_collision_size - len(steps)), value=-100)
        for steps in collision_rewind_steps
    ])

    collision_observations_padded = torch.stack([
        torch.nn.functional.pad(obs, (0, 0, 0, 0, 0, max_collision_size - obs.shape[0]), value=-100)
        for obs in collision_observations
    ])

    collision_actions_padded = torch.stack([
        torch.nn.functional.pad(act, (0, max_collision_size - act.shape[0]), value=-100)
        for act in collision_actions
    ])

    collision_valid_mask = torch.ones(len(batch), max_collision_size, dtype=torch.bool)
    for i, data in enumerate(batch):
        collision_valid_mask[i, data[2].shape[0]:] = False

    return observations, actions, valid_mask, collision_rewind_steps_padded, collision_observations_padded, collision_actions_padded, collision_valid_mask

if __name__ == '__main__':

  data_folder = '/storage/Datasets/highway_env/highway_fast_v0_dqn_meta_action/rollouts'

  dataset = HighwayDataset(data_folder)

  # define the dataloader
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
