import os
import re
import sys

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

nltk.download("stopwords", quiet=True)


def preprocess_text(text):
    """
    Функція для очищення тексту
    - Видалення HTML
    - Видалення спецсимволів (залишаємо тільки літери)
    - Приведення до нижнього регістру
    - Стемінг та видалення стоп-слів
    """
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = text.split()

    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    stop_words.update(["movie", "film", "one", "show", "watch"])

    cleaned_tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return " ".join(cleaned_tokens)


def main():
    if len(sys.argv) != 3:
        print("Помилка. Синтаксис: python src/prepare.py <input_file> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]  # data/raw/dataset.csv
    output_dir = sys.argv[2]  # data/prepared

    df = pd.read_csv(input_file)
    df["original_index"] = df.index
    df["cleaned_review"] = df["review"].apply(preprocess_text)

    df["target"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)

    train_df, test_df = train_test_split(
        df[["cleaned_review", "target", "original_index"]],
        test_size=0.2,
        random_state=42,
    )

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Збережено дані у {output_dir}")


if __name__ == "__main__":
    main()
