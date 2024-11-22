import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

from torch import optim, nn, utils, Tensor
import lightning as L
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append('/u/shuhan/projects/vla')
from src.environments.highway_env.dataset import HighwayDataset, collate_fn
from src.models.vlas.cont_obs_token_action_cot import ContObsTokenActionCOTVLA
from src.auto_labeling.highway_env.lane_change import LaneChangeTaskSpec

torch.set_float32_matmul_precision('high')

# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, loss_weight: dict, cot_cfg: dict, cot_mode: str):
      super().__init__()
      llm_backbone = AutoModelForCausalLM.from_pretrained('gpt2')
      tokenizer = AutoTokenizer.from_pretrained('gpt2')

      hidden_dim = 768
      obs_dim = 25
      num_actions = 5
      mlp_layers = 2

      task_spec_func = LaneChangeTaskSpec


      self.vla = ContObsTokenActionCOTVLA(llm_backbone, tokenizer, task_spec_func, obs_dim, num_actions, hidden_dim, mlp_layers, loss_weight, cot_mode, cot_cfg)
    
    def training_step(self, batch, batch_idx):
       obs, act, valid_mask = batch
       _, loss_dict, batch_preview_strs = self.vla(obs, act, valid_mask)
       loss = loss_dict['total']

        #    action_acc = ((predictions['action'].argmax(dim=-1) == batch[1])[batch[2]]).float().mean()
        #    self.log('action_acc', action_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
       for k, v in loss_dict.items():
          self.log(k + '_loss', v, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
    #    self.log('batch_preview_strs', batch_preview_strs, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)

       return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=156500)
        return [optimizer], [scheduler]
  

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--action_weight', type=float, default=1.0)
    parser.add_argument('--reconst_weight', type=float, default=1.0)
    parser.add_argument('--cot_weight', type=float, default=1.0)
    parser.add_argument('--cot_cls_weight', type=float, default=1.0)
    parser.add_argument('--rollout_stop_weight', type=float, default=1.0)
    parser.add_argument('--cot_mode', type=str, default='all')
    parser.add_argument('--batch_size', type=int, default=28)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()
    
    save_root = '~/results/vla/quick_run_cot'
    save_path = os.path.join(save_root, args.exp_name)
    save_path = os.path.expanduser(save_path)
    os.makedirs(save_path, exist_ok=True)

    loss_weight = {"action": args.action_weight, "obs": 0.0, 'reconst': args.reconst_weight, "cot": args.cot_weight, "cot_cls": args.cot_cls_weight, "rollout_stop": args.rollout_stop_weight}
    cot_cfg = {'lanes_count': 5, 'max_hop': 4, 'cot_index_mode': 'both'}
    cot_mode = args.cot_mode
    
    if args.ckpt_path is not None:
        model = LitAutoEncoder.load_from_checkpoint(os.path.expanduser(args.ckpt_path), loss_weight=loss_weight, cot_cfg=cot_cfg, cot_mode=cot_mode)
    else:
        model = LitAutoEncoder(loss_weight, cot_cfg, cot_mode)

    data_folder = '/storage/Datasets/highway_env/highway_fast_v0_dqn_meta_action_5_lanes/rollouts_train'

    num_gpus = torch.cuda.device_count()

    print(f'Using {num_gpus} GPUs')
    print(f'Saving to {save_path}')

    dataset = HighwayDataset(data_folder, overfit=args.overfit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

    if num_gpus > 0:
        trainer = L.Trainer(max_epochs=args.num_epochs, accelerator='gpu', devices=num_gpus, default_root_dir=save_path, strategy='ddp_find_unused_parameters_true', log_every_n_steps=50)
    else:
        trainer = L.Trainer(max_epochs=args.num_epochs, default_root_dir=save_path, strategy='ddp_find_unused_parameters_true', log_every_n_steps=50)

    trainer.fit(model, train_dataloaders=dataloader)

if __name__ == "__main__":
    main()