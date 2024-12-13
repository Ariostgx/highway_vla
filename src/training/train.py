import os
import sys
import glob
import wandb
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm import tqdm
import logging
from datetime import datetime
from accelerate.utils import LoggerType, ProjectConfiguration

sys.path.append('/u/shuhan/projects/vla')
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.environments.highway_env.dataset import HighwayCollisionDataset, collate_fn_collision, HighwayEnvInitStateDataset, collate_fn_env_init_state
from src.models.vlas.cont_obs_token_action_cot_unified_token_collision import ContObsTokenActionCOTVLAUnifiedTokenCollision
from src.auto_labeling.highway_env.lane_change import LaneChangeTaskSpecCollision
from src.conf.highway_env.vla import get_config
from src.inference.highway_env.rollout import rollout_one_episode


torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Main function
def main():
    # Step 1: Setup configuration
    args, loss_weight, cot_cfg = setup_config()

    save_root = '/storage/Models/shuhan/vla/cot_unified_collision'
    save_path = os.path.join(save_root, args.exp_name)
    save_path = os.path.expanduser(save_path)
    os.makedirs(save_path, exist_ok=True)

    if len(args.ckpt_path) > 0:
        ckpt_path = args.ckpt_path
    else:
        ckpt_folders = glob.glob(os.path.join(save_path, 'checkpoints/*'))
        if len(ckpt_folders) > 0 and not args.always_from_scratch:
            ckpt_path = sorted(ckpt_folders)[-1]
        else:
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
    train_dataloader, rollout_dataloader = setup_dataset(args)

    # Step 5: Initialize model
    model = setup_model(args, loss_weight, cot_cfg)
    # Step 6: Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8, betas=(0.9, 0.98), weight_decay = 1e-1)

    lr_scheduler = get_scheduler(
        "cosine", optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_epochs*len(train_dataloader) / args.gradient_accumulation_steps
    )

    # Step 7: Prepare everything with Accelerator
    model, optimizer, train_dataloader, rollout_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, rollout_dataloader, lr_scheduler)

    if ckpt_path is not None:
        print(f'Loading checkpoint from {ckpt_path}')
        try:
            accelerator.load_state(ckpt_path)
        except Exception as e:
            print(f'Failed to load checkpoint from {ckpt_path}, error: {e}')
            print(f'training from scratch')
    else:
        print('No checkpoint found, starting from scratch')

    # Step 8: Train the model
    if args.rollout_only:
        step = 0
        rollout(model, step, rollout_dataloader, accelerator, args)
    else:
        train(model, optimizer, lr_scheduler, train_dataloader, rollout_dataloader, accelerator, args, save_path)

    accelerator.end_training()
    if accelerator.is_main_process:
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

def setup_dataset(args):
    data_folder = '/storage/Datasets/highway_env/highway_fast_v0_dqn_meta_action_5_lanes/rollouts_train_collision'
    dataset = HighwayCollisionDataset(data_folder, overfit=args.overfit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_collision, num_workers=args.num_workers)

    # rollout_data_path = '/storage/Datasets/highway_env/highway_fast_v0_dqn_meta_action_5_lanes/rollouts_env/rollouts_env_states_8_sample.pkl'
    rollout_data_path = '/storage/Datasets/highway_env/highway_fast_v0_dqn_meta_action_5_lanes/rollouts_env/rollouts_env_states_1k.pkl'
    rollout_dataset = HighwayEnvInitStateDataset(rollout_data_path)
    rollout_dataloader = DataLoader(rollout_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_env_init_state, num_workers=args.num_workers)

    return dataloader, rollout_dataloader

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
def train(model, optimizer, lr_scheduler, train_dataloader, rollout_dataloader, accelerator, args, save_path):
    step = 0
    best_collision_rate = 1.0
    best_collision_step = 0

    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)

    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            outputs = model(batch)

            loss_dict = outputs[0]
            loss = loss_dict['total']

            accelerator.backward(scaler.scale(loss))

            if (step+1) % args.gradient_accumulation_steps == 0:
                if args.loss_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.loss_clip)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                scaler.step(optimizer)
                scaler.update()

            step += 1

            progress_bar.set_postfix(loss=loss.item())

            # Save checkpoint periodically
            if step % args.save_steps == 0:
                accelerator.save_state()

            if step % args.log_freq == 0 and accelerator.is_main_process:
                for k, v in loss_dict.items():
                    wandb.log({f'train/loss_{k}': v}, step=step)
                    wandb.log({f'train/lr': optimizer.param_groups[0]['lr']}, step=step)
            
            if step % args.rollout_steps == 0:
                rollout_collision_rate = rollout(model, step, rollout_dataloader, accelerator, args)
                if rollout_collision_rate < best_collision_rate:
                    if accelerator.is_main_process:
                        print(f'New best collision rate: {rollout_collision_rate}, step: {step}')
    
                        prev_best_path = os.path.join(save_path, f'checkpoints/checkpoint_best_collision_step_{best_collision_step}')
                        best_model_path = os.path.join(save_path, f'checkpoints/checkpoint_best_collision_step_{step}')
                        
                        print(f'Saving checkpoint to {best_model_path}')
                        os.makedirs(best_model_path, exist_ok=True)
                        model_state_dict = model.state_dict()
                        torch.save(model_state_dict, best_model_path+'/model.pth')

                        if os.path.exists(prev_best_path):
                            os.remove(prev_best_path)

                    best_collision_rate = rollout_collision_rate
                    best_collision_step = step

def rollout(model, step, rollout_dataloader, accelerator, args):
    model.eval()
    machine_id = accelerator.local_process_index
    progress_bar = tqdm(rollout_dataloader, desc=f"Rollout at process {machine_id}")

    score_names = ['exact_match_score', 'subset_coverage', 'rollout_collision', 'model_failed', 'action_count', 'token_count', 'collision_detect_recall', 'rewind_precision', 'rewind_collision_avoid_rate', 'model_rewind_ratio', 'reached_goal', 'exceeded_length']

    # collect scores for model and env
    wm_modes = ['model', 'env']
    all_mode_scores = {mode: {k: [] for k in score_names} for mode in wm_modes}
    with torch.no_grad():
        for env_state_cache, path_info in rollout_dataloader:
            for mode in wm_modes:
                # only support batch-size 1 rollout for now
                scores = rollout_one_episode(model.module, env_state_cache[0], path_info[0], use_wm=True, wm_mode=mode, cot_mode='pred', max_rewind_step=args.max_rewind_step)
                for k, v in scores.items():
                    if v is not None:
                        all_mode_scores[mode][k].append(v)
            progress_bar.update(1)
        progress_bar.close()

    # Convert scores to tensors and move to the device
    local_mean_scores = {
        mode: {
            k: torch.tensor(np.mean(v), device=accelerator.device) if len(v) > 0 else torch.tensor(0.0, device=accelerator.device)
            for k, v in all_mode_scores[mode].items()
        }
        for mode in wm_modes
    }

    # print(f'machine_id: {machine_id}, local_mean_scores: {local_mean_scores}')

    # Gather all scores across devices
    global_scores = {mode: {k: [] for k in score_names} for mode in wm_modes}
    for mode in wm_modes:
        for k in score_names:
            gathered_scores = accelerator.gather(local_mean_scores[mode][k])
            global_scores[mode][k] = gathered_scores.cpu()

    # print(f'machine_id: {machine_id}, global_scores: {global_scores}')
    # print('machine_id: ', machine_id, 'finished gathering')
    
    if accelerator.is_main_process:
        # print('machine_id: ', machine_id, 'start logging')
        for mode in wm_modes:
            for k, v in global_scores[mode].items():
                wandb.log({f'rollout_{mode}_pred/{k}': torch.mean(v).item()}, step=step)
    

    # print('machine_id: ', machine_id, 'finished logging')

    # # return the rollout collision rate for checkpointing
    return global_scores['env']['rollout_collision'].mean().item()
            

# Run the script
if __name__ == "__main__":
    main()
