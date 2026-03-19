## Запустити і зберегти експеримент

dvc repro
git add .
git commit -m "run HPO experiment"

## Відтворити експеримент

git checkout old_experiment_commit_id(наприклад a1b2c3d)
dvc repro