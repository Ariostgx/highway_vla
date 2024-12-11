import os
import sys
import glob
import wandb
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import Adam
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm import tqdm
import logging
from datetime import datetime
from accelerate.utils import LoggerType, ProjectConfiguration

sys.path.append('/u/shuhan/projects/vla')
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.environments.highway_env.dataset import HighwayCollisionDataset, collate_fn_collision
from src.models.vlas.cont_obs_token_action_cot_unified_token_collision import ContObsTokenActionCOTVLAUnifiedTokenCollision
from src.auto_labeling.highway_env.lane_change import LaneChangeTaskSpecCollision
from src.conf.highway_env.vla import get_config

torch.set_float32_matmul_precision('high')

# Main function
def main():
    # Step 1: Setup configuration
    args, loss_weight, cot_cfg = setup_config()

    save_root = '/storage/Models/shuhan/vla/cot_unified_collision'
    save_path = os.path.join(save_root, args.exp_name)
    save_path = os.path.expanduser(save_path)
    os.makedirs(save_path, exist_ok=True)

    ckpt_folders = glob.glob(os.path.join(save_path, 'checkpoints/*'))
    if len(ckpt_folders) > 0 and not args.always_from_scratch:
        ckpt_path = sorted(ckpt_folders)[-1]
        print(f'Loading checkpoint from {ckpt_path}')
    else:
        print('No checkpoint found, starting from scratch')
        ckpt_path = None
    
    # # Step 2: Initialize Accelerator
    project_config = ProjectConfiguration(project_dir=save_path, automatic_checkpoint_naming=True, 
    total_limit=1)
    mixed_precision = 'fp16' if args.fp16 else 'no'
    accelerator = Accelerator(mixed_precision=mixed_precision, project_config=project_config)
    accelerator.init_trackers(project_name='vla_exp', config=vars(args))
    with open(os.path.join(save_path, 'configs.json'), 'w') as f:
        json.dump(vars(args), f)

    # Step 3: Setup logging
    if accelerator.is_main_process:
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(project='vla_highwayenv', name=f'{args.exp_name}_{time_str}', config=vars(args))

    # Step 4: Load dataset and dataloader
    train_dataloader = setup_dataset(args)

    # Step 5: Initialize model
    model = setup_model(args, loss_weight, cot_cfg)
    # Step 6: Setup optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_step)

    # Step 7: Prepare everything with Accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    if ckpt_path is not None:
        accelerator.load_state(ckpt_path)

    # Step 8: Train the model
    train(model, optimizer, lr_scheduler, train_dataloader, accelerator, args)

    accelerator.end_training()
    wandb.finish()

# Setup configuration
def setup_config():
    args = get_config()

    loss_weight = {"action": args.action_weight, "obs": 0.0, 'reconst': args.reconst_weight, "cot": args.cot_weight, "separator": args.separator_weight, "rollout_stop": args.rollout_stop_weight, "wm": args.wm_weight}

    cot_cfg = {'lanes_count': 5, 'max_hop': 4, 'cot_index_mode': 'both', 'action_sample_mode': args.action_sample_mode, 'safe_reflect_rate': args.safe_reflect_rate, 'collide_reflect_rate': args.collide_reflect_rate, 'collide_rewind_rate': args.collide_rewind_rate, 'max_rewind_step': args.max_rewind_step, 'shortest_seq_rate': args.shortest_seq_rate}

    return args, loss_weight, cot_cfg
    

# Setup logging
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

# Dataset
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 3, 224, 224)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def setup_dataset(args):
    data_folder = '/storage/Datasets/highway_env/highway_fast_v0_dqn_meta_action_5_lanes/rollouts_train_collision'
    dataset = HighwayCollisionDataset(data_folder, overfit=args.overfit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_collision, num_workers=args.num_workers)

    return dataloader

# Model
def setup_model(args, loss_weight, cot_cfg):
    cot_mode = 'all'
    llm_model = args.llm_model


    cached_path = f'/storage/Models/shuhan/llms/{llm_model}'
    llm_backbone = AutoModelForCausalLM.from_pretrained(cached_path)
    tokenizer = AutoTokenizer.from_pretrained(cached_path)

    if llm_model == 'gpt2':
        hidden_dim = 768
    elif llm_model == 'SmolLM2-135M-Instruct':
        hidden_dim = 576
    elif llm_model == 'SmolLM2-360M-Instruct':
        hidden_dim = 960
    else:
        raise ValueError(f'Unknown LLM model: {llm_model}')
      
    obs_dim = 25
    num_actions = 5
    mlp_layers = 2

    task_spec_func = LaneChangeTaskSpecCollision

    print(f'mask_collision_action: {args.mask_collision_action}')

    model = ContObsTokenActionCOTVLAUnifiedTokenCollision(llm_backbone, tokenizer, task_spec_func, obs_dim, num_actions, hidden_dim, mlp_layers, loss_weight, cot_mode, cot_cfg, max_obs_len=50, max_token_num=args.max_token_num, use_wm=args.use_wm, mask_collision_action=args.mask_collision_action)

    if args.torch_compile:
        model = torch.compile(model)

    return model

# Training loop
def train(model, optimizer, lr_scheduler, train_dataloader, accelerator, args):
    step = 0
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            outputs = model(batch)

            loss_dict = outputs[0]
            loss = loss_dict['total']

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            lr_scheduler.step()
            step += 1

            progress_bar.set_postfix(loss=loss.item())

            # Save checkpoint periodically
            if step % args.save_steps == 0:
                accelerator.save_state()
            
            if step % args.log_freq == 0 and accelerator.is_main_process:
                wandb.log(loss_dict, step=step)

# Run the script
if __name__ == "__main__":
    main()
