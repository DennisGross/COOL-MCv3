#mlflow run safe_training/ --env-manager=local -P num_episodes=102
python cool_mc.py --num_episodes 102 --project_name="taxi_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism"
