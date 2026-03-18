import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_processed_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    abs_path = to_absolute_path(path)
    if abs_path.endswith((".pkl", ".pickle")):
        obj = joblib.load(abs_path)
        if isinstance(obj, dict):
            if {"X_train", "X_test", "y_train", "y_test"}.issubset(obj.keys()):
                return obj["X_train"], obj["X_test"], obj["y_train"], obj["y_test"]
            if {"X", "y"}.issubset(obj.keys()):
                X = obj["X"]
                y = obj["y"]
            else:
                raise ValueError("Unknown format ({X,y} or {X_train,X_test,y_train,y_test}).")
        elif isinstance(obj, pd.DataFrame):
            if "target" not in obj.columns:
                raise ValueError("DataFrame should contains 'target' column.")
            X = obj.drop(columns=["target"]).values
            y = obj["target"].values
        else:
            raise ValueError("Unsupported format (dict or pandas.DataFrame).")
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        return X_train, X_test, y_train, y_test

    if abs_path.endswith(".csv"):
        df = pd.read_csv(abs_path)
        if "target" not in df.columns:
            raise ValueError("CSV should contains 'target' column.")
        X = df.drop(columns=["target"]).values
        y = df["target"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        return X_train, X_test, y_train, y_test

    raise ValueError("Supporting - .pickle/.pkl або .csv.")

def build_model(model_type: str, params: Dict[str, Any], seed: int) -> Any:
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if model_type == "logistic_regression":
        clf = LogisticRegression(random_state=seed, max_iter=500, **params)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    raise ValueError(f"Unknown model.type='{model_type}'. Expecting 'random_forest' or 'logistic_regression'.")

def evaluate(model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, metric: str) -> float:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if metric == "f1":
        return float(f1_score(y_test, y_pred, average="binary" if len(np.unique(y_test)) == 2 else "weighted"))
    if metric == "roc_auc":
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        if len(np.unique(y_test)) > 2:
            return float(roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="weighted"))
        return float(roc_auc_score(y_test, y_score))
    raise ValueError("Unsupported metrics. Use 'f1' or 'roc_auc'.")

def evaluate_cv(model: Any, X: np.ndarray, y: np.ndarray, metric: str, seed: int, n_splits: int = 5) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        m = clone(model)
        scores.append(evaluate(m, X_tr, y_tr, X_te, y_te, metric))
    return float(np.mean(scores))

def make_sampler(sampler_name: str, seed: int, grid_space: Dict[str, Any] = None) -> optuna.samplers.BaseSampler:
    sampler_name = sampler_name.lower()
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler_name == "grid":
        if not grid_space:
            raise ValueError("For sampler='grid' need to set grid_space.")
        return optuna.samplers.GridSampler(search_space=grid_space)
    raise ValueError("sampler should be: tpe, random, grid")

def suggest_params(trial: optuna.Trial, model_type: str, cfg: DictConfig) -> Dict[str, Any]:
    if model_type == "random_forest":
        space = cfg.hpo.random_forest
        return {
            "n_estimators": trial.suggest_int("n_estimators", space.n_estimators.low, space.n_estimators.high),
            "max_depth": trial.suggest_int("max_depth", space.max_depth.low, space.max_depth.high),
            "min_samples_split": trial.suggest_int("min_samples_split", space.min_samples_split.low, space.min_samples_split.high),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", space.min_samples_leaf.low, space.min_samples_leaf.high),
        }
    if model_type == "logistic_regression":
        space = cfg.hpo.logistic_regression
        return {
            "C": trial.suggest_float("C", space.C.low, space.C.high, log=True),
            "solver": trial.suggest_categorical("solver", list(space.solver)),
            "penalty": trial.suggest_categorical("penalty", list(space.penalty)),
        }
    raise ValueError(f"Unknown model.type='{model_type}'.")

def objective_factory(cfg: DictConfig, X_train, X_test, y_train, y_test):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg.model.type, cfg)
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", cfg.seed)
            mlflow.log_params(params)
            model = build_model(cfg.model.type, params=params, seed=cfg.seed)
            if cfg.hpo.use_cv:
                X = np.concatenate([X_train, X_test], axis=0)
                y = np.concatenate([y_train, y_test], axis=0)
                score = evaluate_cv(model, X, y, metric=cfg.hpo.metric, seed=cfg.seed, n_splits=cfg.hpo.cv_folds)
            else:
                score = evaluate(model, X_train, y_train, X_test, y_test, metric=cfg.hpo.metric)
            mlflow.log_metric(cfg.hpo.metric, score)
            return score
    return objective

def register_model_if_enabled(model_uri: str, model_name: str, stage: str) -> None:
    client = mlflow.tracking.MlflowClient()
    mv = mlflow.register_model(model_uri, model_name)
    client.transition_model_version_stage(name=model_name, version=mv.version, stage=stage)
    client.set_model_version_tag(model_name, mv.version, "registered_by", "lab3")
    client.set_model_version_tag(model_name, mv.version, "stage", stage)

def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    X_train, X_test, y_train, y_test = load_processed_data(cfg.data.processed_path)
    
    grid_space = None
    if cfg.hpo.sampler.lower() == "grid":
        if cfg.model.type == "random_forest":
            grid_space = {
                "n_estimators": list(cfg.hpo.grid.random_forest.n_estimators),
                "max_depth": list(cfg.hpo.grid.random_forest.max_depth),
                "min_samples_split": list(cfg.hpo.grid.random_forest.min_samples_split),
                "min_samples_leaf": list(cfg.hpo.grid.random_forest.min_samples_leaf),
            }
        elif cfg.model.type == "logistic_regression":
            grid_space = {
                "C": list(cfg.hpo.grid.logistic_regression.C),
                "solver": list(cfg.hpo.grid.logistic_regression.solver),
                "penalty": list(cfg.hpo.grid.logistic_regression.penalty),
            }
            
    sampler = make_sampler(cfg.hpo.sampler, seed=cfg.seed, grid_space=grid_space)
    
    with mlflow.start_run(run_name="hpo_parent") as parent_run:
        mlflow.set_tag("model_type", cfg.model.type)
        mlflow.set_tag("sampler", cfg.hpo.sampler)
        mlflow.set_tag("seed", cfg.seed)
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config_resolved.json")
        
        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)
        objective = objective_factory(cfg, X_train, X_test, y_train, y_test)
        study.optimize(objective, n_trials=cfg.hpo.n_trials)
        
        best_trial = study.best_trial
        mlflow.log_metric(f"best_{cfg.hpo.metric}", float(best_trial.value))
        mlflow.log_dict(best_trial.params, "best_params.json")
        
        best_model = build_model(cfg.model.type, params=best_trial.params, seed=cfg.seed)
        best_score = evaluate(best_model, X_train, y_train, X_test, y_test, metric=cfg.hpo.metric)
        mlflow.log_metric(f"final_{cfg.hpo.metric}", best_score)
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.pkl")
        mlflow.log_artifact("models/best_model.pkl")
        
        if cfg.mlflow.log_model:
            import mlflow.sklearn
            mlflow.sklearn.log_model(best_model, artifact_path="model")
            
        if cfg.mlflow.register_model:
            model_uri = f"runs:/{parent_run.info.run_id}/model"
            register_model_if_enabled(model_uri, cfg.mlflow.model_name, stage=cfg.mlflow.stage)

import hydra

@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)

if __name__ == "__main__":
    hydra_entry()