import os
import sys
import time
import hydra
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from datetime import datetime
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments.highway_env.dataset import HighwayCollisionDataset, collate_fn_collision, HighwayEnvInitStateDataset, collate_fn_env_init_state
from src.models.vlas.cont_obs_token_action_cot_unified_token_collision import ContObsTokenActionCOTVLAUnifiedTokenCollision
from src.auto_labeling.highway_env.lane_change import LaneChangeTaskSpecCollision
from src.inference.highway_env.rollout import rollout_one_episode

torch.set_float32_matmul_precision('high')

class VLAUnifiedModel(nn.Module):
    def __init__(self, loss_weight: dict, cot_cfg: dict, cot_mode: str, llm_model: str, use_wm: bool, mask_collision_action: bool, T_step: int):
        super().__init__()
        self.T_step = T_step
        
        # Load LLM backbone and tokenizer
        llm_cache_dir = '/storage/Models/shuhan/llms/SmolLM2-135M-Instruct'
        llm_backbone = AutoModelForCausalLM.from_pretrained(llm_cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(llm_cache_dir)

        # Set hidden dimension based on model
        if llm_model == 'gpt2':
            hidden_dim = 768
        elif llm_model == 'HuggingFaceTB/SmolLM2-135M-Instruct':
            hidden_dim = 576
        elif llm_model == 'HuggingFaceTB/SmolLM2-360M-Instruct':
            hidden_dim = 960
        else:
            raise ValueError(f'Unknown LLM model: {llm_model}')
        
        # Model parameters
        obs_dim = 25
        num_actions = 5
        mlp_layers = 2
        task_spec_func = LaneChangeTaskSpecCollision

        print(f'mask_collision_action: {mask_collision_action}')

        # Create VLA model
        self.vla = ContObsTokenActionCOTVLAUnifiedTokenCollision(
            llm_backbone, 
            tokenizer, 
            task_spec_func, 
            obs_dim, 
            num_actions, 
            hidden_dim, 
            mlp_layers, 
            loss_weight, 
            cot_mode, 
            cot_cfg, 
            max_obs_len=50, 
            use_wm=use_wm, 
            mask_collision_action=mask_collision_action
        )

    def forward(self, batch):
        return self.vla(batch)

def rollout_evaluation(model, rollout_dataloader, rank, world_size, step, max_rewind_step):
    """
    Perform rollout evaluation in a distributed setting
    """
    model.eval()
    
    score_names = ['exact_match_score', 'subset_coverage', 'rollout_collision', 'model_failed', 
                   'action_count', 'token_count', 'collision_detect_recall', 'rewind_precision', 
                   'rewind_collision_avoid_rate', 'model_rewind_ratio', 'reached_goal', 'exceeded_length']

    # collect scores for model and env
    wm_modes = ['model', 'env']
    all_mode_scores = {mode: {k: [] for k in score_names} for mode in wm_modes}
    
    if rank == 0:
        progress_bar = tqdm(rollout_dataloader, desc=f"Rollout Evaluation")
    
    with torch.no_grad():
        for env_state_cache, path_info in rollout_dataloader:
            for mode in wm_modes:
                # only support batch-size 1 rollout for now
                scores = rollout_one_episode(
                    model.module.vla if hasattr(model, 'module') else model.vla, 
                    env_state_cache[0], 
                    path_info[0], 
                    use_wm=True, 
                    wm_mode=mode, 
                    cot_mode='pred', 
                    max_rewind_step=max_rewind_step
                )
                for k, v in scores.items():
                    if v is not None:
                        all_mode_scores[mode][k].append(v)
            
            if rank == 0:
                progress_bar.update(1)
    
    if rank == 0:
        progress_bar.close()

    # Convert scores to tensors and move to the device
    device = next(model.parameters()).device
    local_mean_scores = {
        mode: {
            k: torch.tensor(np.mean(v), device=device) if len(v) > 0 else torch.tensor(0.0, device=device)
            for k, v in all_mode_scores[mode].items()
        }
        for mode in wm_modes
    }

    # Gather all scores across devices
    global_scores = {mode: {k: [] for k in score_names} for mode in wm_modes}
    for mode in wm_modes:
        for k in score_names:
            # Simple all-gather implementation
            gathered_scores = [torch.zeros_like(local_mean_scores[mode][k]) for _ in range(world_size)]
            dist.all_gather(gathered_scores, local_mean_scores[mode][k])
            global_scores[mode][k] = torch.stack(gathered_scores)

    # Log to wandb (only on rank 0)
    if rank == 0:
        for mode in wm_modes:
            for k, v in global_scores[mode].items():
                wandb.log({f'rollout_{mode}_pred/{k}': torch.mean(v).item()}, step=step)
        
        # Print some key metrics
        print(f"Rollout Evaluation Step {step}:")
        for mode in wm_modes:
            collision_rate = torch.mean(global_scores[mode]['rollout_collision']).item()
            success_rate = torch.mean(global_scores[mode]['reached_goal']).item()
            print(f"  {mode.upper()} - Collision Rate: {collision_rate:.3f}, Success Rate: {success_rate:.3f}")

    # Return the rollout collision rate for checkpointing
    return torch.mean(global_scores['env']['rollout_collision']).item()

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # Convert config to dict for easier access
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    
    # Get distributed info
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.cuda.device_count()
    device = f'cuda:{device_id}'
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f'World size: {world_size}, Device: {device}')
    
    # Setup paths
    save_root = os.path.expanduser(cfg_dict['save_root'])
    save_path = os.path.join(save_root, cfg_dict['exp_name'])
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
        print(f'Saving to {save_path}')

    # Initialize wandb (only on rank 0)
    if rank == 0:
        # Create a unique run name with timestamp if not specified in config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg_dict['exp_name']}_{timestamp}"
        
        # Initialize wandb
        wandb.init(
            project=cfg_dict['wandb']['project'],
            name=run_name,
            config=cfg_dict,  # Log the entire Hydra config
            dir=save_path,    # Save wandb files in the experiment directory
        )
        
        # Log the config file
        config_path = os.path.join(save_path, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(config=cfg, f=f)
        wandb.save(config_path)

    # Setup model configuration
    loss_weight = {
        "action": cfg_dict['action_weight'],
        "obs": 0.0,
        'reconst': cfg_dict['reconst_weight'],
        "cot": cfg_dict['cot_weight'],
        "separator": cfg_dict['separator_weight'],
        "rollout_stop": cfg_dict['rollout_stop_weight'],
        "wm": cfg_dict['wm_weight']
    }

    cot_cfg = {
        'lanes_count': 5,
        'max_hop': 4,
        'cot_index_mode': 'both',
        'action_sample_mode': cfg_dict['action_sample_mode'],
        'safe_reflect_rate': cfg_dict['safe_reflect_rate'],
        'collide_reflect_rate': cfg_dict['collide_reflect_rate'],
        'collide_rewind_rate': cfg_dict['collide_rewind_rate'],
        'max_rewind_step': cfg_dict['max_rewind_step'],
        'shortest_seq_rate': cfg_dict['shortest_seq_rate']
    }
    cot_mode = 'all'

    # Create model and move to device
    model = VLAUnifiedModel(
        loss_weight,
        cot_cfg,
        cot_mode,
        cfg_dict['llm_model'],
        use_wm=cfg_dict['use_wm'],
        mask_collision_action=cfg_dict['mask_collision_action'],
        T_step=cfg_dict['T_step']
    ).to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[device_id])

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_dict['T_step'])

    # Setup training dataset and dataloader
    train_dataset = HighwayCollisionDataset(cfg_dict['data_folder'], overfit=cfg_dict['overfit'])
    
    # Create distributed sampler for training
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg_dict['batch_size'],
        sampler=train_sampler,
        shuffle=False,  # Shuffling is handled by the sampler
        collate_fn=collate_fn_collision,
        num_workers=4,
        pin_memory=True
    )

    # Setup rollout evaluation dataset and dataloader
    rollout_data_path = cfg_dict.get('rollout_data_path', '/storage/Datasets/highway_env/highway_fast_v0_dqn_meta_action_5_lanes/rollouts_env/rollouts_env_states_1k.pkl')
    rollout_dataset = HighwayEnvInitStateDataset(rollout_data_path)
    
    # Create distributed sampler for rollout evaluation
    rollout_sampler = DistributedSampler(
        rollout_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False  # No need to shuffle for evaluation
    )
    
    rollout_dataloader = DataLoader(
        rollout_dataset,
        batch_size=1,  # Rollout only supports batch size 1
        sampler=rollout_sampler,
        shuffle=False,
        collate_fn=collate_fn_env_init_state,
        num_workers=2,
        pin_memory=True
    )

    # Training loop
    total_steps = cfg_dict['num_epochs'] * len(train_dataloader)
    global_step = 0
    
    if rank == 0:
        print(f'Starting training for {total_steps} steps')
        print(f'Training dataset size: {len(train_dataset)}')
        print(f'Rollout dataset size: {len(rollout_dataset)}')
    
    for epoch in range(cfg_dict['num_epochs']):
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch}', disable=rank != 0):
            # Forward pass
            model_return = model(batch)
            loss_dict = model_return[0]
            loss = loss_dict['total']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Logging
            if rank == 0:
                # Log to wandb every 50 steps
                if global_step % 50 == 0:
                    # Prepare metrics for logging
                    metrics = {
                        'train/total_loss': loss.item(),
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'train/epoch': epoch,
                        'train/global_step': global_step,
                    }
                    # Add individual loss components
                    for k, v in loss_dict.items():
                        metrics[f'train/{k}_loss'] = v.item()
                    
                    # Log to wandb
                    wandb.log(metrics)
                    
                    # Print to console
                    print(f'Step {global_step}, Loss: {loss.item():.4f}')
                    for k, v in loss_dict.items():
                        print(f'{k}: {v.item():.4f}')
                    print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
            
            global_step += 1
            epoch_loss += loss.item()
        
        # End of epoch
        if rank == 0:
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f'Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}')
            
            # Log epoch metrics to wandb
            wandb.log({
                'epoch/avg_loss': avg_epoch_loss,
                'epoch/epoch': epoch,
            })

        Perform rollout evaluation every 5 epochs
        if (epoch + 1) % 5 == 0:
            if rank == 0:
                print(f'Starting rollout evaluation at epoch {epoch}...')
            
            rollout_collision_rate = rollout_evaluation(
                model, 
                rollout_dataloader, 
                rank, 
                world_size, 
                global_step,
                cfg_dict['max_rewind_step']
            )
            
            if rank == 0:
                print(f'Rollout evaluation completed. Collision rate: {rollout_collision_rate:.3f}')
        
        # Save checkpoint (only on rank 0)
        if rank == 0 and (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
            }
            checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            
            # Log checkpoint to wandb
            wandb.save(checkpoint_path)

    # Cleanup
    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main() 