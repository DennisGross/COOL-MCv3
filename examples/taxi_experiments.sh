python cool_mc.py --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --layers=4 --neurons=512 --lr=0.0001 --batch_size=32 --num_episodes=100000 --eval_interval=100 --epsilon_dec=0.99999 --epsilon_min=0.1 --gamma=0.99 --epsilon=1 --replace=301 --reward_flag=0 --wrong_action_penalty=0 --prop="Pmax=? [F jobs_done=2]" --max_steps=30
