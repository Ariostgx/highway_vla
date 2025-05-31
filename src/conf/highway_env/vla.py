import argparse
from typing import Self, Type

EXPERIMENT_REGISTRY = {}

def register_experiment(cls):
    """
    Decorator to register an experiment class in the registry.
    """
    if not hasattr(cls, "name") or cls.name is None:
        raise ValueError(f"Experiment class {cls.__name__} must define a 'name' attribute.")
    if cls.name in EXPERIMENT_REGISTRY:
        raise ValueError(f"Experiment name '{cls.name}' is already registered.")
    EXPERIMENT_REGISTRY[cls.name] = cls
    return cls


class BaseExperiment:
    """
    Base configuration class with default values. Supports extension for different experiments.
    """
    name = None  # Experiments must define a name

    DEFAULTS = {
        'action_weight': 1.0,
        'reconst_weight': 1.0,
        'cot_weight': 1.0,
        'separator_weight': 1.0,
        'rollout_stop_weight': 1.0,
        'wm_weight': 1.0,

        'action_sample_mode': 'future',  # 'random', 'future'
        'safe_reflect_rate': 0.2,
        'collide_reflect_rate': 0.8,
        'collide_rewind_rate': 0.8,
        'max_rewind_step': 1,
        'shortest_seq_rate': 0.0,

        'use_wm': False,
        'exp_name': 'test',
        'llm_model': 'SmolLM2-135M-Instruct',  # 'gpt2', 'SmolLM2-135M-Instruct'
        'overfit': False,
        'mask_collision_action': False,
        'ckpt_path': "",

        'lr': 1e-3,
        'lr_scheduler': 'cosine',
        'batch_size': 12,
        'gradient_accumulation_steps': 1,
        'num_epochs': 5,
        'warmup_steps': 100,
        'num_workers': 4,
        'save_steps': 10000,
        'rollout_steps': 20000,
        'max_token_num': 512,
        'log_freq': 20,
        'loss_clip': 1.0,
        'optimizer': 'adamw',

        'single_gpu': False,
        'fp16': False,
        'torch_compile': False,
        'always_from_scratch': False,
        'rollout_only': False,
    }

    def __init__(self):
        for key, value in self.DEFAULTS.items():
            setattr(self, key, value)

    @classmethod
    def from_args(cls, args):
      """
      Create a configuration instance from command-line arguments.
      Automatically handles subclass-specific defaults.
      """
      parser = argparse.ArgumentParser()

      # Gather defaults from the current class and all parent classes
      all_defaults = {}
      for base in cls.__mro__[::-1]:  # Traverse the method resolution order (MRO)
          if hasattr(base, 'DEFAULTS'):
              all_defaults.update(base.DEFAULTS)

      # Add arguments dynamically
      for key, value in all_defaults.items():
          parser.add_argument(f'--{key}', type=type(value), default=value)

      parsed_args = parser.parse_args(args)  # Parse the arguments properly

      # Create an instance of the class and set attributes
      config = cls()
      for key, value in vars(parsed_args).items():
          setattr(config, key, value)

      return config

    def __repr__(self):
        """
        String representation of the configuration for easy debugging.
        """
        config_attrs = {key: getattr(self, key) for key in vars(self)}
        class_name = self.__class__.__name__
        return f"{class_name}({config_attrs})"


@register_experiment
class WithWMModel(BaseExperiment):
    name = 'with_wm'

    NEW_DEFAULTS = {
        'use_wm': True,
    }

    DEFAULTS = {**BaseExperiment.DEFAULTS, **NEW_DEFAULTS}


@register_experiment
class WithWMRewind4(WithWMModel):
    name = 'with_wm_rewind_4'

    NEW_DEFAULTS = {
        'max_rewind_step': 4,
        'action_sample_mode': 'random',
    }

    DEFAULTS = {**WithWMModel.DEFAULTS, **NEW_DEFAULTS}


def get_config():
    initial_parser = argparse.ArgumentParser()
    initial_parser.add_argument('--base_cfg', type=str, required=True, help="Name of the experiment class to use")
    initial_args, remaining_args = initial_parser.parse_known_args()

    # Retrieve the corresponding experiment class
    if initial_args.base_cfg not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Experiment name '{initial_args.base_cfg}' is not registered.")

    experiment_class = EXPERIMENT_REGISTRY[initial_args.base_cfg]
    config = experiment_class.from_args(remaining_args)

    print(config)

    return config

# Main execution
if __name__ == "__main__":
    config = get_config()
    print(config.use_wm)
    print(config)

