accelerate launch --config_file /u/shuhan/projects/vla/scripts/accl_train/4gpu.yaml /u/shuhan/projects/vla/src/training/train.py --base_cfg with_wm_rewind_4 --exp_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_4_grad_accum_1 --batch_size 12 --gradient_accumulation_steps 1

accelerate launch --config_file /u/shuhan/projects/vla/scripts/accl_train/4gpu.yaml /u/shuhan/projects/vla/src/training/train.py --base_cfg with_wm_rewind_4 --exp_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_4_grad_accum_10 --batch_size 12 --gradient_accumulation_steps 10

accelerate launch --config_file /u/shuhan/projects/vla/scripts/accl_train/4gpu.yaml /u/shuhan/projects/vla/src/training/train.py --base_cfg with_wm_rewind_4 --exp_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_4_grad_accum_5 --batch_size 12 --gradient_accumulation_steps 5

accelerate launch --config_file /u/shuhan/projects/vla/scripts/accl_train/4gpu.yaml /u/shuhan/projects/vla/src/training/train.py --base_cfg with_wm_rewind_4 --exp_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_4_grad_accum_2 --batch_size 12 --gradient_accumulation_steps 2