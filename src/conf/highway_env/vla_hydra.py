import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any


class HydraConfigWrapper:
    """
    Wrapper class to maintain compatibility with the original config system.
    Converts Hydra DictConfig to a class with attribute access.
    """
    
    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        # Set all config values as attributes for backward compatibility
        for key, value in cfg.items():
            setattr(self, key, value)
    
    def __repr__(self):
        """String representation of the configuration for easy debugging."""
        config_attrs = {key: getattr(self, key) for key in vars(self) if not key.startswith('_')}
        return f"HydraConfig({config_attrs})"
    
    def get_hydra_cfg(self) -> DictConfig:
        """Get the original Hydra DictConfig object."""
        return self._cfg


def create_config_from_hydra(cfg: DictConfig) -> HydraConfigWrapper:
    """
    Create a config wrapper from Hydra configuration.
    
    Args:
        cfg: Hydra DictConfig
        
    Returns:
        HydraConfigWrapper: Wrapped configuration with attribute access
    """
    return HydraConfigWrapper(cfg)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function that loads configuration using Hydra.
    This replaces the original get_config() function from vla.py
    
    Args:
        cfg: Configuration loaded by Hydra
    """
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(cfg))
    
    # Create a wrapper for backward compatibility
    config = create_config_from_hydra(cfg)
    
    # Print some example attributes
    print(f"use_wm: {config.use_wm}")
    print(f"exp_name: {config.exp_name}")
    print(f"lr: {config.lr}")
    print(f"batch_size: {config.batch_size}")
    print("\nFull config:")
    print(config)
    
    return cfg


# Example usage functions
def example_usage():
    """
    Example of how to use the Hydra configuration system.
    
    To run different experiments:
    
    Base experiment:
    python vla_hydra.py
    
    With WM experiment:
    python vla_hydra.py experiment=with_wm
    
    With WM Rewind 4 experiment:
    python vla_hydra.py experiment=with_wm_rewind_4
    
    Override specific parameters:
    python vla_hydra.py experiment=with_wm lr=1e-4 batch_size=16
    
    Multiple overrides:
    python vla_hydra.py experiment=with_wm_rewind_4 lr=2e-4 num_epochs=10 use_wm=false
    """
    pass


if __name__ == "__main__":
    # This will automatically load the configuration based on command line arguments
    main() 