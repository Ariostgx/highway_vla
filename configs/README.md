# Hydra Configuration System

This directory contains the Hydra configuration files that replace the original Python-based configuration system from `src/conf/highway_env/vla.py`.

## Structure

```
configs/
├── config.yaml                    # Main config file defining defaults
├── experiment/                    # Experiment configurations
│   ├── base.yaml                 # Base experiment (equivalent to BaseExperiment)
│   ├── with_wm.yaml             # With World Model (equivalent to WithWMModel)
│   └── with_wm_rewind_4.yaml    # With WM and rewind=4 (equivalent to WithWMRewind4)
└── README.md                     # This file
```

## Usage

### Basic Usage

Run with base configuration:
```bash
python src/conf/highway_env/vla_hydra.py
```

### Experiment Selection

Run specific experiments:
```bash
# Base experiment
python src/conf/highway_env/vla_hydra.py experiment=base

# With World Model
python src/conf/highway_env/vla_hydra.py experiment=with_wm

# With World Model and rewind=4
python src/conf/highway_env/vla_hydra.py experiment=with_wm_rewind_4
```

### Parameter Overrides

Override specific parameters:
```bash
# Change learning rate
python src/conf/highway_env/vla_hydra.py experiment=with_wm lr=1e-4

# Multiple overrides
python src/conf/highway_env/vla_hydra.py experiment=with_wm lr=2e-4 batch_size=16 num_epochs=10

# Override boolean values
python src/conf/highway_env/vla_hydra.py experiment=with_wm use_wm=false fp16=true
```

### Configuration Composition

Hydra allows you to compose configurations. The hierarchy is:
1. Base experiment configuration (`experiment/base.yaml`)
2. Specific experiment overrides (e.g., `experiment/with_wm.yaml`)
3. Command-line overrides

## Migration from Original System

The original Python classes map to Hydra configs as follows:

| Original Class | Hydra Config | Description |
|---|---|---|
| `BaseExperiment` | `experiment/base.yaml` | Base configuration with all defaults |
| `WithWMModel` | `experiment/with_wm.yaml` | Enables world model (`use_wm: true`) |
| `WithWMRewind4` | `experiment/with_wm_rewind_4.yaml` | Adds rewind=4 and random sampling |

## Configuration Parameters

All parameters from the original `BaseExperiment.DEFAULTS` are available:

### Model Weights
- `action_weight`: 1.0
- `reconst_weight`: 1.0
- `cot_weight`: 1.0
- `separator_weight`: 1.0
- `rollout_stop_weight`: 1.0
- `wm_weight`: 1.0

### Training Configuration
- `action_sample_mode`: 'future' or 'random'
- `safe_reflect_rate`: 0.2
- `collide_reflect_rate`: 0.8
- `collide_rewind_rate`: 0.8
- `max_rewind_step`: 1
- `shortest_seq_rate`: 0.0

### Model Settings
- `use_wm`: false
- `exp_name`: 'test'
- `llm_model`: 'SmolLM2-135M-Instruct'
- `overfit`: false
- `mask_collision_action`: false
- `ckpt_path`: ""

### Training Hyperparameters
- `lr`: 1e-3
- `lr_scheduler`: 'cosine'
- `batch_size`: 12
- `gradient_accumulation_steps`: 1
- `num_epochs`: 5
- `warmup_steps`: 100
- `num_workers`: 4
- `save_steps`: 10000
- `rollout_steps`: 20000
- `max_token_num`: 512
- `log_freq`: 20
- `loss_clip`: 1.0
- `optimizer`: 'adamw'

### Training Control
- `single_gpu`: false
- `fp16`: false
- `torch_compile`: false
- `always_from_scratch`: false
- `rollout_only`: false

## Benefits of Hydra

1. **Cleaner Separation**: Configuration is separated from code
2. **Easy Experimentation**: Switch between experiments with a single parameter
3. **Powerful Overrides**: Override any parameter from command line
4. **Configuration Composition**: Inherit and override configurations
5. **Automatic Logging**: Hydra automatically logs configurations
6. **Type Safety**: Built-in validation and type checking
7. **Multirun Support**: Easy hyperparameter sweeps

## Example Integration

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # Use cfg.parameter_name to access configuration
    print(f"Learning rate: {cfg.lr}")
    print(f"Use WM: {cfg.use_wm}")
    
    # For backward compatibility, you can use the wrapper:
    from src.conf.highway_env.vla_hydra import create_config_from_hydra
    config = create_config_from_hydra(cfg)
    print(f"Batch size: {config.batch_size}")

if __name__ == "__main__":
    my_app()
``` 