# import argparse
# import json
# import os
# import sys

# import matplotlib.pyplot as plt
# import mlflow
# import mlflow.sklearn
# import pandas as pd
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
# from sklearn.pipeline import Pipeline

# EXPERIMENT_NAME = "IMDB_Sentiment_Analysis"


# def plot_feature_importance(pipeline, top_n=20):
#     feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
#     coefs = pipeline.named_steps["clf"].coef_.flatten()
#     # print(coefs)

#     df_importance = pd.DataFrame({"feature": feature_names, "importance": coefs})

#     top_positive = df_importance.nlargest(top_n, "importance")
#     top_negative = df_importance.nsmallest(top_n, "importance")

#     df_plot = pd.concat([top_positive, top_negative])

#     plt.figure(figsize=(10, 8))
#     colors = ["red" if x < 0 else "green" for x in df_plot["importance"]]
#     sns.barplot(x="importance", y="feature", data=df_plot, palette=colors)
#     plt.title(f"Top {top_n} Positive & Negative Words")
#     plt.xlabel("Coefficient Magnitude")
#     plt.tight_layout()

#     filename = "feature_importance.png"
#     plt.savefig(filename)
#     plt.close()
#     return filename


# def main(args):
#     mlflow.set_experiment(EXPERIMENT_NAME)

#     with mlflow.start_run():
#         mlflow.set_tag("author", "Duda Artem")
#         mlflow.set_tag("model_type", "LogisticRegression")
#         mlflow.set_tag("dataset_version", "imdb_v1")

#         mlflow.log_params(vars(args))

#         train_path = os.path.join(args.input_dir, "train.csv")
#         test_path = os.path.join(args.input_dir, "test.csv")

#         train_df = pd.read_csv(train_path)
#         test_df = pd.read_csv(test_path)

#         if "original_index" in train_df.columns:
#             split_indices = {
#                 "train_indices": train_df["original_index"].tolist(),
#                 "test_indices": test_df["original_index"].tolist(),
#             }
#             indices_filename = "split_indices.json"
#             with open(indices_filename, "w") as f:
#                 json.dump(split_indices, f)

#             mlflow.log_artifact(indices_filename)
#         else:
#             print("Ознаки 'original_index' не знайдено")
#             sys.exit(1)

#         X_train = train_df["cleaned_review"]
#         y_train = train_df["target"]
#         X_test = test_df["cleaned_review"]
#         y_test = test_df["target"]

#         pipeline = Pipeline(
#             [
#                 (
#                     "tfidf",
#                     TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 1)),
#                 ),
#                 ("clf", LogisticRegression(C=args.C, max_iter=500)),
#             ]
#         )

#         pipeline.fit(X_train, y_train)

#         y_pred = pipeline.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)

#         print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.log_metric("f1_score", f1)

#         mlflow.sklearn.log_model(pipeline, "model")

#         cm = confusion_matrix(y_test, y_pred)
#         plt.figure(figsize=(6, 5))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#         plt.title("Confusion Matrix")
#         plt.savefig("confusion_matrix.png")
#         plt.close()
#         mlflow.log_artifact("confusion_matrix.png")

#         fi_plot_path = plot_feature_importance(pipeline)
#         mlflow.log_artifact(fi_plot_path)


# if __name__ == "__main__":
#     print("hello")
#     parser = argparse.ArgumentParser(description="Train Sentiment Analysis Model")

#     parser.add_argument(
#         "--input_dir",
#         type=str,
#         default="data/prepared",
#         help="директорія з підготовленими даними (train.csv та test.csv)",
#     )

#     parser.add_argument(
#         "--C",
#         type=float,
#         default=1.0,
#         help="значення регуляризації логістичної регресії",
#     )
#     parser.add_argument(
#         "--max_features",
#         type=int,
#         default=5000,
#         help="макс. кількість ознак для TF-IDF",
#     )

#     args = parser.parse_args()

#     main(args)
