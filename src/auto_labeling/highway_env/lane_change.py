import numpy as np

from .base import HighwayEnvTaskSpec

class LaneChangeTaskSpec(HighwayEnvTaskSpec):
  def __init__(self, observations: np.ndarray, actions: np.ndarray, cfgs: dict):
    super().__init__(observations, actions, cfgs)
    
    assert 'lanes_count' in cfgs, "lanes_count must be specified"
    assert 'max_hop' in cfgs, "max_hop must be specified"
    assert 'cot_index_mode' in cfgs, "cot_index_mode must be specified"

    self.hop_lane_ids, self.hop_indices = self._get_task_lane_ids()
  
  def get_task_hop_info(self) -> dict:
    return {
      'hop_lane_ids': self.hop_lane_ids,
      'hop_indices': self.hop_indices
    }
  
  def _compute_ego_lane_ids(self, obs: np.ndarray) -> np.ndarray:
    """
    Compute the lane ids of the ego vehicle.
    """
    lane_cnt = self.cfgs['lanes_count']

    lane_width = 1.0 / lane_cnt

    abs_y = obs[..., 2].copy()
    abs_y[:, 1:] += abs_y[:, :1]
    abs_y += lane_width / 2

    lane_ids = (abs_y / lane_width).astype(np.int32)

    return lane_ids[:, 0]

  def _get_lane_change_timesteps(self, lane_ids: np.ndarray) -> np.ndarray:
    """
    Get the timesteps when the ego vehicle changes lanes.
    """
    return np.where(np.diff(lane_ids) != 0)[0]
  
  def _get_task_lane_ids(self) -> np.ndarray:
    """
    Get the lane ids of the task.
    """
    ego_lane_ids = self._compute_ego_lane_ids(self.observations)
    
    max_hop = self.cfgs['max_hop']
    ego_hop_indices = self._get_lane_change_timesteps(ego_lane_ids)[:max_hop]

    hop_lane_ids = [ego_lane_ids[0].item()] + ego_lane_ids[ego_hop_indices+1].tolist()
    hop_indices = np.concatenate([[0], ego_hop_indices])

    return hop_lane_ids, hop_indices
    
  def get_goal_spec(self) -> str:
    goal = f"Lane {self.hop_lane_ids[-1]}"
    path = " -> ".join([f"Lane {lane_id}" for lane_id in self.hop_lane_ids])
    
    return f"Goal is to reach {goal}. Need to go through path {path}."

  def get_ending_spec(self) -> str:
    return f"Now at Goal Lane {self.hop_lane_ids[-1]}. Finished."

  def get_multi_step_cot_prompt(self) -> dict[int, str]:
    '''
    cot_index_mode:
      - right_before_next_action: cot right before the next effective action. conduct reasoning before executing the new action.
      - right_after_last_action: cot right after the last effective action. conduct reasoning when reaching the new location.
      - both: include both modes. the same cot will be generated at most twice.
      
      for both mode, the cot is placed after the cot_index observation, before the cot_index action
    '''

    cot_prompts = {}
    cot_index_mode = self.cfgs['cot_index_mode']
    
    for idx, hop_idx in enumerate(self.hop_indices[:-1]):
      current_lane_id = self.hop_lane_ids[idx]
      next_lane_id = self.hop_lane_ids[idx+1]
      path = " -> ".join([f"Lane {lane_id}" for lane_id in self.hop_lane_ids[hop_idx+1:-1]])
      action = 'turn right' if next_lane_id > current_lane_id else 'turn left'

      if len(path) > 0:
        path = f"Follow {path} and reach Lane {self.hop_lane_ids[-1]}"
      else:
        path = f"Goal Reachable"

      cot_prompt = f"Now at Lane {current_lane_id}. {path}. Next is Lane {next_lane_id}. Action: {action}."
      
      cot_indices = []
      if cot_index_mode == 'right_before_next_action' or cot_index_mode == 'both':
        cot_indices.append(self.hop_indices[idx+1].item())
      if cot_index_mode == 'right_after_last_action' or cot_index_mode == 'both':
        cot_indices.append(hop_idx.item() + 1 if idx > 0 else 0)
      
      for cot_index in cot_indices:
        cot_prompts[cot_index] = cot_prompt

    return cot_prompts