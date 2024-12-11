# python highway_env_rollout_unified_collision_multistep.py --model_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_2 --wm_mode model --cot_mode pred --max_rewind_step 2
# python highway_env_rollout_unified_collision_multistep.py --model_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_2 --wm_mode model --cot_mode always --max_rewind_step 2

python highway_env_rollout_unified_collision_multistep.py --model_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_4_WM_1e2 --wm_mode model --cot_mode pred --max_rewind_step 4
python highway_env_rollout_unified_collision_multistep.py --model_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_4_WM_1e2 --wm_mode model --cot_mode always --max_rewind_step 4
python highway_env_rollout_unified_collision_multistep.py --model_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_4_WM_1e2 --wm_mode env --cot_mode pred --max_rewind_step 4
python highway_env_rollout_unified_collision_multistep.py --model_name with_wm_cr_0.8_re_0.8_sr_0.2_max_rewind_step_4_WM_1e2 --wm_mode env --cot_mode always --max_rewind_step 4