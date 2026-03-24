import os
import json
import pytest

def test_artifacts_exist():
    artifacts = [
        "models/model.pkl", 
        "reports/metrics.json", 
        "reports/confusion_matrix.png"
    ]
    for artifact in artifacts:
        assert os.path.exists(artifact), f"Артефакт не знайдено: {artifact}"

def test_quality_gate():
    metrics_path = "reports/metrics.json"
    assert os.path.exists(metrics_path), "Файл метрик відсутній"
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "f1" in metrics, "Метрика 'f1' не залогована"
    assert "accuracy" in metrics, "Метрика 'accuracy' не залогована"
    
    f1_score = metrics["f1"]
    
    threshold = float(os.getenv("F1_THRESHOLD", "0.80"))
    
    assert f1_score >= threshold, f"Quality Gate не пройдено: F1 моделі ({f1_score:.4f}) нижчий за мінімальний поріг ({threshold:.2f})"