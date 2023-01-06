python cool_mc.py --num_episodes 3 --project_name="taxi_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --preprocessor="normalizer,10" --num_episodes 102
# By passing an empty preprocessor, we use the previous run's preprocessor
python cool_mc.py --num_episodes 3 --project_name="taxi_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --preprocessor="" --parent_run_id="last"
# The following command uses no preprocessor
python cool_mc.py --num_episodes 3 --project_name="taxi_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --preprocessor="None" --parent_run_id="last"
