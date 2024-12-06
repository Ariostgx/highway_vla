# python train_cont_obs_token_action_cot_vla_unified_collision.py --use_wm --safe_reflect_rate 0.2 --collide_reflect_rate 0.8 --collide_rewind_rate 0.8 --batch_size 8 --max_rewind_step 4 --mask_collision_action --exp_name with_wm_cr_0.8_re_0.8_sr_0.2_mask_collision_act_max_rewind_step_4_360M_WM_1e2  --action_sample_mode random --llm_model HuggingFaceTB/SmolLM2-360M-Instruct --wm_weight 100.0

# python train_cont_obs_token_action_cot_vla_unified_collision.py --use_wm --safe_reflect_rate 0.2 --collide_reflect_rate 0.8 --collide_rewind_rate 0.8 --batch_size 16 --max_rewind_step 4 --mask_collision_action --exp_name with_wm_cr_0.8_re_0.8_sr_0.2_mask_collision_act_max_rewind_step_4_WM_1e2  --action_sample_mode random --wm_weight 100.0

python train_cont_obs_token_action_cot_vla_unified_collision.py --use_wm --safe_reflect_rate 0.2 --collide_reflect_rate 0.8 --collide_rewind_rate 0.8 --batch_size 12 --max_rewind_step 4 --exp_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_4_WM_1e2  --action_sample_mode random --wm_weight 100.0


# python train_cont_obs_token_action_cot_vla_unified_collision.py --use_wm --safe_reflect_rate 0.2 --collide_reflect_rate 0.8 --collide_rewind_rate 0.8 --batch_size 16 --max_rewind_step 2 --exp_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_2  --action_sample_mode random


# python train_cont_obs_token_action_cot_vla_unified_collision.py --use_wm --safe_reflect_rate 0.2 --collide_reflect_rate 0.8 --collide_rewind_rate 0.8 --batch_size 16 --max_rewind_step 4 --exp_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_4 --action_sample_mode random