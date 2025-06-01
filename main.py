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

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments.highway_env.dataset import HighwayCollisionDataset, collate_fn_collision
from src.models.vlas.cont_obs_token_action_cot_unified_token_collision import ContObsTokenActionCOTVLAUnifiedTokenCollision
from src.auto_labeling.highway_env.lane_change import LaneChangeTaskSpecCollision

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

    # Setup dataset and dataloader
    dataset = HighwayCollisionDataset(cfg_dict['data_folder'], overfit=cfg_dict['overfit'])
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        # batch_size=cfg_dict['batch_size'] // world_size,  # Divide batch size by world size
        batch_size=cfg_dict['batch_size'],  # Divide batch size by world size
        sampler=sampler,
        shuffle=False,  # Shuffling is handled by the sampler
        collate_fn=collate_fn_collision,
        num_workers=4,
        pin_memory=True
    )

    # Training loop
    total_steps = cfg_dict['num_epochs'] * len(dataloader)
    global_step = 0
    
    if rank == 0:
        print(f'Starting training for {total_steps} steps')
    
    for epoch in range(cfg_dict['num_epochs']):
        # Set epoch for sampler
        sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f'Epoch {epoch}', disable=rank != 0):
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
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f'Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}')
            
            # Log epoch metrics to wandb
            wandb.log({
                'epoch/avg_loss': avg_epoch_loss,
                'epoch/epoch': epoch,
            })
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:  # Save every 5 epochs
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