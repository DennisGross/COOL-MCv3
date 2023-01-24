#python cool_mc.py --task=safe_training --project_name="avoid_experiments" --rl_algorithm=dqn_agent --prism_file_path="avoid.prism" --constant_definitions="xMax=12,yMax=12,slickness=0.1" --prop="" --num_episodes=103  --reward_flag=1 --seed=128 --epsilon=0.5 --layers=2 --neurons=128 --epsilon_min=0.01  --num_episodes=3647 --eval_interval=100 --epsilon_dec=0.9999 --lr=0.001 --replay_buffer_size=200000 --training_threshold=10000
python cool_mc.py --parent_run_id="9326488d9f324c7cb9ca57525a346c50" --project_name="avoid_experiments" --constant_definitions="xMax=12,yMax=12,slickness=0.1" --prism_dir="../prism_files" --seed=128 --prism_file_path="avoid.prism" --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]"
python cool_mc.py --parent_run_id="9326488d9f324c7cb9ca57525a346c50" --project_name="avoid_experiments" --constant_definitions="xMax=13,yMax=13,slickness=0.1" --prism_dir="../prism_files" --prism_file_path="avoid.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]"
python cool_mc.py --parent_run_id="9326488d9f324c7cb9ca57525a346c50" --project_name="avoid_experiments" --constant_definitions="xMax=14,yMax=14,slickness=0.1" --prism_dir="../prism_files" --prism_file_path="avoid.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]"
