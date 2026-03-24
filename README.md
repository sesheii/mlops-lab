## Запустити і зберегти експеримент
```shell
dvc repro
git add .
git commit -m "run HPO experiment"
```
## Відтворити експеримент
```shell
git checkout old_experiment_commit_id(наприклад a1b2c3d)
dvc repro
```