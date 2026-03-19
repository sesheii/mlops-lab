## Зберегти експеримент

git add .
git commit -m "Prepare for HPO experiment"
uv run src/optimize.py

## Відтворити експеримент

git checkout old_experiment_commit_id(наприклад a1b2c3d) 
dvc checkout
uv run src/optimize.py