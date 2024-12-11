accelerate launch --config_file /u/shuhan/projects/vla/scripts/accl_train/4gpu.yaml /u/shuhan/projects/vla/src/training/train.py --base_cfg with_wm_rewind_4 --exp_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_4_short_seq_freq_0.2 --shortest_seq_rate 0.2 --batch_size 12

accelerate launch --config_file /u/shuhan/projects/vla/scripts/accl_train/4gpu.yaml /u/shuhan/projects/vla/src/training/train.py --base_cfg with_wm_rewind_4 --exp_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_4_short_seq_freq_0.5 --shortest_seq_rate 0.5 --batch_size 12

accelerate launch --config_file /u/shuhan/projects/vla/scripts/accl_train/8gpu.yaml /u/shuhan/projects/vla/src/training/train.py --base_cfg with_wm_rewind_4 --exp_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_4_360M --batch_size 6  --llm_model SmolLM2-360M-Instruct
