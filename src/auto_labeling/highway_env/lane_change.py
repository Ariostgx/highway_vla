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

    lane_ids = (abs_y / lane_width).astype(int)

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
      next_path_lanes = [f"Lane {lane_id}" for lane_id in self.hop_lane_ids[idx+1:]]
      path = " -> ".join(next_path_lanes)
      action = 'turn right' if next_lane_id > current_lane_id else 'turn left'

      if len(next_path_lanes) > 1:
        path = f"Follow {path}"
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

class LaneChangeTaskSpecCollision(LaneChangeTaskSpec):
  def __init__(self, observations: np.ndarray, actions: np.ndarray, collision_observations: np.ndarray, collision_actions: np.ndarray, collision_indices: np.ndarray, cfgs: dict):
    super().__init__(observations, actions, cfgs)
    
    assert 'action_sample_mode' in cfgs, "action_sample_mode must be specified"

    self.collision_observations = collision_observations
    self.collision_actions = collision_actions
    self.collision_tidx  = collision_indices

    self.hop_lane_ids, self.hop_indices = self._get_task_lane_ids()
    self.cot_tidx, self.cot_observations, self.cot_collision_actions = self._get_collision_data()
  
  def _get_collision_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # tidx is the index of the self.observations and self.actions
    # cidx is the index of the self.collision_observations and self.collision_actions
    
    select_tidx = []
    select_cot_observations = []
    select_cot_collision_actions = []

    all_unique_tidx = np.unique(self.collision_tidx)

    action_sample_mode = self.cfgs['action_sample_mode']
    
    for tidx in all_unique_tidx:
      coll_cidx_mask = self.collision_tidx == tidx
      coll_cidxs = np.where(coll_cidx_mask)[0]
      
      if action_sample_mode == 'random':
        cot_cidx = np.random.choice(coll_cidxs, 1, replace=False)
      
      elif action_sample_mode == 'future':
        # choose the action that is the closest to the next gt action in the future
        cot_cidx = -1
        best_tidx_dist = np.inf

        for cidx in coll_cidxs:
          cot_action = self.collision_actions[cidx]
          future_gt_actions = self.actions[tidx+1:].tolist()
          if cot_action not in future_gt_actions:
            continue
          gt_action_dist = future_gt_actions.index(cot_action)
          if gt_action_dist < best_tidx_dist:
            best_tidx_dist = gt_action_dist
            cot_cidx = cidx

        if cot_cidx == -1:
          cot_cidx = np.random.choice(coll_cidxs, 1, replace=False)

      select_tidx.append(tidx)
      select_cot_observations.append(self.collision_observations[cot_cidx])
      select_cot_collision_actions.append(self.collision_actions[cot_cidx].astype(np.int32))

    return select_tidx, select_cot_observations, select_cot_collision_actions