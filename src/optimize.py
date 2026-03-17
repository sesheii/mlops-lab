import os
import json
import random
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
import joblib
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone

import hydra._internal.utils
hydra._internal.utils.LazyCompletionHelp = str
import hydra
from omegaconf import DictConfig, OmegaConf

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def objective_factory(cfg: DictConfig, X, y):
    def objective(trial: optuna.Trial) -> float:
        c_val = trial.suggest_float(
            "C", 
            cfg.hpo.search_space.C.low, 
            cfg.hpo.search_space.C.high, 
            log=True
        )
        max_feat = trial.suggest_int(
            "max_features", 
            cfg.hpo.search_space.max_features.low, 
            cfg.hpo.search_space.max_features.high, 
            step=cfg.hpo.search_space.max_features.step
        )
        ngram_str = trial.suggest_categorical(
            "ngram_range", 
            list(cfg.hpo.search_space.ngram_range)
        )
        ngram_tuple = tuple(map(int, ngram_str.split(",")))

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", cfg.seed)
            
            params = {"C": c_val, "max_features": max_feat, "ngram_range": ngram_str}
            mlflow.log_params(params)

            base_pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=max_feat, ngram_range=ngram_tuple)),
                ("clf", LogisticRegression(C=c_val, max_iter=cfg.model.max_iter, random_state=cfg.seed))
            ])

            cv = StratifiedKFold(n_splits=cfg.hpo.cv_folds, shuffle=True, random_state=cfg.seed)
            fold_scores = []

            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                pipeline = clone(base_pipeline)
                
                pipeline.fit(X_train_fold, y_train_fold)
                y_pred_fold = pipeline.predict(X_val_fold)
                
                fold_score = f1_score(y_val_fold, y_pred_fold)
                fold_scores.append(fold_score)
            
            mean_score = np.mean(fold_scores)
            mlflow.log_metric(f"mean_{cfg.hpo.metric}_cv", mean_score)

            return mean_score
    return objective

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    set_global_seed(cfg.seed)
    
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    df_train = pd.read_csv(cfg.data.train_path).dropna()
    X_train_full = df_train["cleaned_review"].values
    y_train_full = df_train["target"].values

    df_test = pd.read_csv(cfg.data.test_path).dropna()
    X_test = df_test["cleaned_review"].values
    y_test = df_test["target"].values

    if cfg.hpo.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    elif cfg.hpo.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=cfg.seed)
    elif cfg.hpo.sampler == "grid":
        grid_space = {
            "C": list(cfg.hpo.search_space.C),
            "max_features": list(cfg.hpo.search_space.max_features),
            "ngram_range": list(cfg.hpo.search_space.ngram_range)
        }
        sampler = optuna.samplers.GridSampler(search_space=grid_space)
    else:
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)

    with mlflow.start_run(run_name=f"HPO_{cfg.hpo.sampler}") as parent_run:
        mlflow.set_tag("optimization", "optuna")
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.json")

        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)
        
        objective = objective_factory(cfg, X_train_full, y_train_full)
        study.optimize(objective, n_trials=cfg.hpo.n_trials)

        best_trial = study.best_trial

        mlflow.log_metric(f"best_cv_{cfg.hpo.metric}", best_trial.value)
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})

        with open("best_params.json", "w") as f:
            json.dump(best_trial.params, f)
        mlflow.log_artifact("best_params.json")

        best_ngram_tuple = tuple(map(int, best_trial.params["ngram_range"].split(",")))

        best_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=best_trial.params["max_features"], ngram_range=best_ngram_tuple)),
            ("clf", LogisticRegression(C=best_trial.params["C"], max_iter=cfg.model.max_iter, random_state=cfg.seed))
        ])
        
        best_pipeline.fit(X_train_full, y_train_full)
        y_pred_test = best_pipeline.predict(X_test)
        final_test_score = f1_score(y_test, y_pred_test)
        
        mlflow.log_metric(f"final_test_{cfg.hpo.metric}", final_test_score)

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_pipeline, "models/best_model.pkl")
        mlflow.log_artifact("models/best_model.pkl")

        artifact_path = "model"
        mlflow.sklearn.log_model(best_pipeline, artifact_path)
        
        if cfg.mlflow.get("register_model", True):
            model_uri = f"runs:/{parent_run.info.run_id}/{artifact_path}"
            model_name = cfg.mlflow.get("model_name", "IMDB_Sentiment_Model")
            
            mv = mlflow.register_model(model_uri, model_name)
            
            client = MlflowClient()
            stage = cfg.mlflow.get("stage", "Staging")
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage=stage
            )

if __name__ == "__main__":
    main()