# Migration Guide: Old Config System → Hydra

This guide shows how to migrate from the old Python-based configuration system to the new Hydra-based system.

## Command Line Interface Changes

### Old System
```bash
python src/training/train.py --base_cfg with_wm_rewind_4 --exp_name my_experiment --batch_size 16 --lr 1e-4
```

### New Hydra System
```bash
python src/training/train.py experiment=with_wm_rewind_4 exp_name=my_experiment batch_size=16 lr=1e-4
```

## Key Differences

| Old System | New Hydra System | Notes |
|------------|------------------|-------|
| `--base_cfg EXPERIMENT` | `experiment=EXPERIMENT` | Select experiment configuration |
| `--param_name value` | `param_name=value` | No double dashes, use equals sign |
| `--flag` (boolean) | `flag=true/false` | Must specify true/false explicitly |

## Available Experiments

| Experiment Name | Description |
|----------------|-------------|
| `base` | Base configuration (equivalent to `BaseExperiment`) |
| `with_wm` | Enable world model (equivalent to `WithWMModel`) |
| `with_wm_rewind_4` | World model with rewind=4 (equivalent to `WithWMRewind4`) |

## Example Conversions

### Training Scripts

**Old:**
```bash
accelerate launch --config_file scripts/accl_train/4gpu.yaml src/training/train.py \
    --base_cfg with_wm_rewind_4 \
    --exp_name my_experiment \
    --batch_size 12 \
    --gradient_accumulation_steps 5 \
    --lr 2e-4 \
    --num_epochs 10
```

**New:**
```bash
accelerate launch --config_file scripts/accl_train/4gpu.yaml src/training/train.py \
    experiment=with_wm_rewind_4 \
    exp_name=my_experiment \
    batch_size=12 \
    gradient_accumulation_steps=5 \
    lr=2e-4 \
    num_epochs=10
```

### Boolean Parameters

**Old:**
```bash
python src/training/train.py --base_cfg base --fp16 --torch_compile
```

**New:**
```bash
python src/training/train.py experiment=base fp16=true torch_compile=true
```

## Benefits of New System

1. **Configuration Files**: Experiments are now defined in YAML files under `configs/experiment/`
2. **Composition**: Easy to inherit and override configurations
3. **Type Safety**: Automatic validation and type checking
4. **Better Logging**: Hydra automatically tracks configuration changes
5. **Reproducibility**: Configuration is automatically saved with each run
6. **Flexibility**: Easy parameter sweeps and multirun support

## Advanced Usage

### Parameter Sweeps (Multirun)
```bash
python src/training/train.py --multirun experiment=with_wm lr=1e-4,2e-4,5e-4 batch_size=8,12,16
```

### Override Multiple Parameters
```bash
python src/training/train.py experiment=with_wm lr=1e-4 batch_size=16 num_epochs=20 use_wm=false
```

### Working Directory
Hydra automatically creates output directories with timestamps. To specify a custom working directory:
```bash
python src/training/train.py experiment=base hydra.run.dir=./my_custom_output
``` 