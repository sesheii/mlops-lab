import random
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import subprocess
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

import hydra._internal.utils

hydra._internal.utils.LazyCompletionHelp = str
import hydra
from omegaconf import DictConfig, OmegaConf


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_git_commit():
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        return commit
    except Exception:
        return "unknown"


def objective_factory(cfg: DictConfig, X, y):
    def objective(trial: optuna.Trial) -> float:
        if cfg.hpo.sampler == "grid":
            max_feat = trial.suggest_categorical(
                "max_features", list(cfg.hpo.search_space.tfidf.max_features)
            )
        else:
            max_feat = trial.suggest_int(
                "max_features",
                cfg.hpo.search_space.tfidf.max_features.low,
                cfg.hpo.search_space.tfidf.max_features.high,
                step=cfg.hpo.search_space.tfidf.max_features.step,
            )
        ngram_str = trial.suggest_categorical(
            "ngram_range", list(cfg.hpo.search_space.tfidf.ngram_range)
        )
        ngram_tuple = tuple(map(int, ngram_str.split(",")))

        params = {"max_features": max_feat, "ngram_range": ngram_str}

        if cfg.model.type == "logistic_regression":
            if cfg.hpo.sampler == "grid":
                c_val = trial.suggest_categorical(
                    "C", list(cfg.hpo.search_space.logistic_regression.C)
                )
            else:
                c_val = trial.suggest_float(
                    "C",
                    cfg.hpo.search_space.logistic_regression.C.low,
                    cfg.hpo.search_space.logistic_regression.C.high,
                    log=True,
                )
            params["C"] = c_val
            clf = LogisticRegression(
                C=c_val, max_iter=cfg.model.get("max_iter", 500), random_state=cfg.seed
            )

        elif cfg.model.type == "random_forest":
            if cfg.hpo.sampler == "grid":
                n_est = trial.suggest_categorical(
                    "n_estimators",
                    list(cfg.hpo.search_space.random_forest.n_estimators),
                )
                m_depth = trial.suggest_categorical(
                    "max_depth", list(cfg.hpo.search_space.random_forest.max_depth)
                )
            else:
                n_est = trial.suggest_int(
                    "n_estimators",
                    cfg.hpo.search_space.random_forest.n_estimators.low,
                    cfg.hpo.search_space.random_forest.n_estimators.high,
                )
                m_depth = trial.suggest_int(
                    "max_depth",
                    cfg.hpo.search_space.random_forest.max_depth.low,
                    cfg.hpo.search_space.random_forest.max_depth.high,
                )
            params["n_estimators"] = n_est
            params["max_depth"] = m_depth
            clf = RandomForestClassifier(
                n_estimators=n_est, max_depth=m_depth, random_state=cfg.seed, n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model.type: {cfg.model.type}")

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", cfg.seed)

            mlflow.log_params(params)

            base_pipeline = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(max_features=max_feat, ngram_range=ngram_tuple),
                    ),
                    ("clf", clf),
                ]
            )

            cv = StratifiedKFold(
                n_splits=cfg.hpo.cv_folds, shuffle=True, random_state=cfg.seed
            )
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

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    is_ci = os.getenv("CI", "false").lower() == "true"

    if is_ci:
        n_trials = 2
        sample_size = 1000
    else:
        n_trials = cfg.hpo.n_trials
        sample_size = None

    df_train = pd.read_csv(cfg.data.train_path).dropna()
    df_test = pd.read_csv(cfg.data.test_path).dropna()
    if sample_size:
        df_train = df_train.sample(
            n=min(sample_size, len(df_train)), random_state=cfg.seed
        )
        df_test = df_test.sample(
            n=min(sample_size // 2, len(df_test)), random_state=cfg.seed
        )

    X_train_full = df_train["cleaned_review"].values
    y_train_full = df_train["target"].values

    X_test = df_test["cleaned_review"].values
    y_test = df_test["target"].values

    if cfg.hpo.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    elif cfg.hpo.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=cfg.seed)
    elif cfg.hpo.sampler == "grid":
        grid_space = {
            "max_features": list(cfg.hpo.search_space.tfidf.max_features),
            "ngram_range": list(cfg.hpo.search_space.tfidf.ngram_range),
        }
        if cfg.model.type == "logistic_regression":
            grid_space["C"] = list(cfg.hpo.search_space.logistic_regression.C)
        elif cfg.model.type == "random_forest":
            grid_space["n_estimators"] = list(
                cfg.hpo.search_space.random_forest.n_estimators
            )
            grid_space["max_depth"] = list(cfg.hpo.search_space.random_forest.max_depth)

        sampler = optuna.samplers.GridSampler(search_space=grid_space)
    else:
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)

    with mlflow.start_run(
        run_name=f"HPO_{cfg.hpo.sampler}_{cfg.model.type}"
    ) as parent_run:
        mlflow.set_tag("optimization", "optuna")
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.json")

        commit_hash = get_git_commit()
        mlflow.set_tag("mlflow.source.git.commit", commit_hash)

        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)

        objective = objective_factory(cfg, X_train_full, y_train_full)
        study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial

        mlflow.log_metric(f"best_cv_{cfg.hpo.metric}", best_trial.value)
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})

        mlflow.log_dict(best_trial.params, "best_params.json")

        best_ngram_tuple = tuple(map(int, best_trial.params["ngram_range"].split(",")))

        if cfg.model.type == "logistic_regression":
            best_clf = LogisticRegression(
                C=best_trial.params["C"],
                max_iter=cfg.model.get("max_iter", 500),
                random_state=cfg.seed,
            )
        elif cfg.model.type == "random_forest":
            best_clf = RandomForestClassifier(
                n_estimators=best_trial.params["n_estimators"],
                max_depth=best_trial.params["max_depth"],
                random_state=cfg.seed,
                n_jobs=-1,
            )

        best_pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=best_trial.params["max_features"],
                        ngram_range=best_ngram_tuple,
                    ),
                ),
                ("clf", best_clf),
            ]
        )

        best_pipeline.fit(X_train_full, y_train_full)
        y_pred_test = best_pipeline.predict(X_test)
        final_test_f1 = f1_score(y_test, y_pred_test)
        final_test_acc = accuracy_score(y_test, y_pred_test)

        mlflow.log_metric(f"final_test_{cfg.hpo.metric}", final_test_f1)
        mlflow.log_metric("final_test_accuracy", final_test_acc)

        if os.path.exists("pyproject.toml"):
            mlflow.log_artifact("pyproject.toml")

        artifact_path = "model"
        mlflow.sklearn.log_model(sk_model=best_pipeline, artifact_path="model", registered_model_name="IMDB_RF_Model")

        if cfg.mlflow.get("register_model", False):
            model_uri = f"runs:/{parent_run.info.run_id}/{artifact_path}"

            model_name = cfg.mlflow.model_name

            mv = mlflow.register_model(model_uri, model_name)

            client = MlflowClient()
            stage = cfg.mlflow.get("stage", "Staging")
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage=stage
            )

        os.makedirs("reports", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_pipeline, "models/model.pkl")
        metrics_dict = {"accuracy": float(final_test_acc), "f1": float(final_test_f1)}
        with open("reports/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig("reports/confusion_matrix.png")
        plt.close()


if __name__ == "__main__":
    main()
