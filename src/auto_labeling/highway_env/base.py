import gymnasium as gym
import numpy as np

from highway_env.utils import lmap
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.kinematics import Vehicle

class HighwayEnvTaskSpec:
  '''
  Base class for specifying tasks in the highway environment given observations and actions of a single rollout.
  '''
  def __init__(self, observations: np.ndarray, actions: np.ndarray, cfgs: dict):
    '''
    Args:
      observations: np.ndarray, shape (N, obs_dim)
      actions: np.ndarray, shape (N, act_dim)
      cfgs: dict, additional configurations
    Records the observations and actions of a single rollout.
    Potentially conduct useful preprocessing here.
    '''
    self.observations = observations
    self.actions = actions
    self.cfgs = cfgs

  def get_goal_spec(self) -> str:
    '''
    Returns a string description of the goal of the task.
    This goal specification will normally be used as the task specification for the sequence.
    '''
    raise NotImplementedError

  def get_multi_step_cot_prompt(self) -> dict[int, str]:
    '''
    Returns a dictionary of multi step CoT prompts.
    The keys are the number of steps ahead to look, and the values are the CoT prompts at that step.
    Only key timesteps that are needed for a CoT reasoning prompt will be returned.
    '''
    raise NotImplementedError

  def get_ending_spec(self) -> str:
    '''
    Returns a string description of the ending of the task.
    '''
    raise NotImplementedError