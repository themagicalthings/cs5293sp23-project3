import os
import warnings
import logging

import nltk
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Suppress warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger("PyPDF2").setLevel(logging.ERROR)

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("wordnet")


def read_pdf(file_path):
    pdf = PdfReader(open(file_path, "rb"))
    text = " ".join([page.extract_text() for page in pdf.pages])
    return text


def preprocess_text(text, extra_stopwords):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    words = [word for word in words if word not in stopwords.words("english") + extra_stopwords]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = f"Topic #{topic_idx}: "
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)


def main():
    # Define the folder containing the PDF files
    pdf_folder = "smartcity/"

    # Add your extra stopwords here
    extra_stopwords = ["city", "state", "smart", "page"]

    # Initialize the DataFrame to store results
    data = []
    # Iterate through all PDF files in the folder
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file_name)

            # Read the PDF file
            raw_text = read_pdf(file_path)
            clean_text = preprocess_text(raw_text, extra_stopwords)
            data.append([os.path.basename(file_path).split(".")[0], raw_text, clean_text])

    # Create a DataFrame with the preprocessed data
    df = pd.DataFrame(data, columns=["city", "raw text", "clean text"])

    # Prepare the data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean text"])
    X_dense = X.toarray()

    # Initialize variables
    k_values = list(range(2, 51))
    results = {
        "K-means": {},
        "Hierarchical": {},
        "DBSCAN": {},
    }

    optimal_k = {
        "K-means": {"k": 0, "silhouette_score": -1},
        "Hierarchical": {"k": 0, "silhouette_score": -1},
    }

    # Evaluate clustering models
    for k in k_values:
        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)
        kmeans_silhouette = silhouette_score(X, kmeans_labels)
        kmeans_calinski_harabasz = calinski_harabasz_score(X_dense, kmeans_labels)
        kmeans_davies_bouldin = davies_bouldin_score(X_dense, kmeans_labels)
        results["K-means"][k] = [kmeans_silhouette, kmeans_calinski_harabasz, kmeans_davies_bouldin]

        if kmeans_silhouette > optimal_k["K-means"]["silhouette_score"]:
            optimal_k["K-means"]["k"] = k
            optimal_k["K-means"]["silhouette_score"] = kmeans_silhouette

        # Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=k)
        hierarchical_labels = hierarchical.fit_predict(X_dense)
        hierarchical_silhouette = silhouette_score(X, hierarchical_labels)
        hierarchical_calinski_harabasz = calinski_harabasz_score(X_dense, hierarchical_labels)
        hierarchical_davies_bouldin = davies_bouldin_score(X_dense, hierarchical_labels)
        results["Hierarchical"][k] = [hierarchical_silhouette, hierarchical_calinski_harabasz, hierarchical_davies_bouldin]

        if hierarchical_silhouette > optimal_k["Hierarchical"]["silhouette_score"]:
            optimal_k["Hierarchical"]["k"] = k
            optimal_k["Hierarchical"]["silhouette_score"] = hierarchical_silhouette

    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    unique_labels = np.unique(dbscan_labels)

    if len(unique_labels) > 1:
        dbscan_silhouette = silhouette_score(X, dbscan_labels)
        dbscan_calinski_harabasz = calinski_harabasz_score(X_dense, dbscan_labels)
        dbscan_davies_bouldin = davies_bouldin_score(X_dense, dbscan_labels)
        results["DBSCAN"] = {"X": [dbscan_silhouette, dbscan_calinski_harabasz, dbscan_davies_bouldin]}
    else:
        results["DBSCAN"] = {"X": ["Not enough clusters", "Not enough clusters", "Not enough clusters"]}

    print(results)

    # First, use the optimal_k values to refit the K-means and Hierarchical models
    kmeans_optimal = KMeans(n_clusters=optimal_k["K-means"]["k"], random_state=42)
    kmeans_optimal_labels = kmeans_optimal.fit_predict(X)

    # Add the cluster labels to the DataFrame
    df["K-means Cluster ID"] = kmeans_optimal_labels

    # Fit the LDA model
    lda = LatentDirichletAllocation(n_components=optimal_k["K-means"]["k"], random_state=0)
    lda.fit(X)

    # Print the top 5 words for each topic
    print_top_words(lda, vectorizer.get_feature_names_out(), 5)

    # Obtain the topic distribution for each city
    topic_distribution = lda.transform(X)

    # Add the topic distributions to the DataFrame
    df_topics = pd.DataFrame(topic_distribution, columns=[f"Topic {i}" for i in range(optimal_k["K-means"]["k"])])
    df = pd.concat([df, df_topics], axis=1)

    # Save the DataFrame to a CSV file
    df.to_csv("smartcity_predict.tsv", index=False)


if __name__ == "__main__":
    main()


