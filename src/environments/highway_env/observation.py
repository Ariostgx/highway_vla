import gymnasium as gym
import numpy as np

from highway_env.utils import lmap
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.kinematics import Vehicle

def rescale(env: gym.Env, obs: np.ndarray) -> np.ndarray:
  feat_names = env.observation_type.features
  side_lanes =env.unwrapped.road.network.all_side_lanes(env.observation_type.observer_vehicle.lane_index)
  features_range = {
                  "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                  "y": [
                      -AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                      AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                  ],
                  "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                  "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
              }
  
  rescaled_obs = np.zeros_like(obs)
  for dim_i, feat_name in enumerate(feat_names):
      if feat_name in features_range: 
          x_range = features_range[feat_name]
          rescaled_obs[..., dim_i] = lmap(obs[..., dim_i], [-1, 1], x_range)
      else:
          rescaled_obs[..., dim_i] = obs[..., dim_i]
  return rescaled_obs