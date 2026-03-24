import os
import pandas as pd
import pytest

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

def test_processed_data_exists():
    assert os.path.exists(TRAIN_PATH), f"Train файл відсутній: {TRAIN_PATH}"
    assert os.path.exists(TEST_PATH), f"Test файл відсутній: {TEST_PATH}"

def test_data_schema():
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    
    required_columns = {"cleaned_review", "target"}
    
    assert required_columns.issubset(df_train.columns), "У train датасеті бракує необхідних колонок"
    assert required_columns.issubset(df_test.columns), "У test датасеті бракує необхідних колонок"

def test_no_missing_values():
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    
    assert df_train.isnull().sum().sum() == 0, "Train датасет містить порожні значення (NaN)"
    assert df_test.isnull().sum().sum() == 0, "Test датасет містить порожні значення (NaN)"

def test_target_classes():
    df_train = pd.read_csv(TRAIN_PATH)
    
    unique_classes = df_train["target"].nunique()
    assert unique_classes == 2, f"Очікувалось 2 класи у 'target', але знайдено: {unique_classes}"
    
    class_counts = df_train["target"].value_counts(normalize=True)
    assert class_counts.min() > 0.1, "Сильний дисбаланс класів (менше 10% для одного з класів)"