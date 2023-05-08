# 
# Project 3: The Smart City Slicker
Welcome to Project 3: The Smart City Slicker! In this project, we will dive into the world of Smart Cities by analyzing and visualizing data from the 2015 Smart City Challenge. As a stakeholder in a rising Smart City, you will learn more about themes and concepts that revolve around existing smart cities and discover where your city stands among others.
Objective

Our goal is to perform exploratory data analysis (EDA) on a dataset from the 2015 Smart City Challenge to uncover facts and insights that can be communicated through text analysis and visualizations. The project will involve data preprocessing or cleaning as necessary to facilitate our understanding of the data.
Structure

The project is divided into two main parts:

    Part 1 - Data Exploration and Cleaning: In this part, you will explore the dataset and clean it to make it suitable for further analysis.
    Part 2 - Modeling and Visualization: This part involves creating models and visualizations using the preprocessed data to extract valuable insights.

Throughout the notebook, you will encounter empty code cells and markdown cells with [Your Answer Here]. You are encouraged to edit and add as many cells as needed to complete the project.
Skills and Tools

During this project, you will apply your knowledge in the following areas:

    Data Cleaning
    Exploratory Data Analysis (EDA)
    Machine Learning
    Data Visualization
    Databases

Getting Started

To get started with the project, follow these steps:

    Clone the repository or download the project files.
    Open the Jupyter Notebook provided in the project folder.
    Follow the instructions and guidelines provided in the notebook.
    Complete the tasks in each section, and document your findings and insights.

By the end of this project, you will have gained valuable experience in data analysis, visualization, and machine learning, which can be applied to various real-world scenarios. Good luck and have fun exploring the world of Smart Cities!

Introduction

In this project, we'll analyze a dataset containing the 2015 Smart City Challenge Applicants (non-finalist) provided by the U.S Department of Transportation. The dataset consists of PDF files for each applicant city, which we'll process and analyze to answer various questions about the applicants and their proposed ideas.

Some of the questions we can explore are:

    Can we identify frequently occurring words that could be removed during data preprocessing?
    Where are the applicants from?
    Are there multiple entries for the same city in different applications?
    What are the major themes and concepts from the smart city applicants?

Loading and Handling Files

To load and process the data, we'll use the pypdf.pdf.PdfFileReader class. The data is stored in the smartcity/ directory, and we'll only need to handle PDF files.

To install the necessary module, use the following command:

pipenv install pypdf

Cleaning Up PDFs

We'll preprocess the data to clean it up and make it more readable. To do this, we'll use code from Text Analytics with Python:

    contractions.py (Pages 136-137)
    text_normalizer.py (Pages 155-156)

In addition to the data cleaning provided by the textbook, we will need to:

    Remove terms that may affect clustering and topic modeling. Consider removing city names, states, and common words (e.g., smart, city, page).
    Check the data to remove applicants whose text was not processed correctly. Do not remove more than 15 cities from the data.

Code Overview

The provided code in the project consists of several steps and functions to load, preprocess, and analyze the data. Here is a more detailed overview of the code structure:

    Importing necessary libraries: The code begins by importing the required libraries, including PyPDF2, pandas, NumPy, NLTK, and various functions from the scikit-learn package. This ensures that all the necessary tools and functions are available to process and analyze the data.

    Defining functions to read and preprocess PDF files:
        read_pdf(file_path): This function takes a file path as input, reads the PDF file using the PdfReader class from the PyPDF2 library, and extracts the text from each page. The extracted texts from all pages are then concatenated and returned as a single string.
        preprocess_text(text, extra_stopwords): This function takes a string and a list of extra stopwords as input, and performs several preprocessing steps. These include converting the text to lowercase, tokenizing the words, removing non-alphabetic words and stopwords, and lemmatizing the words. The preprocessed words are then joined together and returned as a single string.

    Reading and preprocessing data from the smartcity/ directory: The code iterates through each file in the smartcity/ directory, processes the PDF files using the read_pdf() function, and preprocesses the extracted text using the preprocess_text() function. The city name, raw text, and preprocessed text are then appended to the data list.

    Removing cities with issues: During the preprocessing step, the code checks whether the cleaned text is above a specified threshold (20 characters in this case) and limits the number of removed cities to 15. If the cleaned text does not meet these criteria, the city is removed from the dataset, and the corresponding issue is stored in the removed_cities list.

    Creating a pandas DataFrame to store the data: After processing and filtering the data, it is stored in a pandas DataFrame with three columns: "city", "raw text", and "clean text". This structured format allows for easy data manipulation and analysis in subsequent steps.

With the data loaded and preprocessed in a pandas DataFrame, you can now proceed with your analysis to answer the questions posed earlier, perform clustering or topic modeling, and explore other interesting aspects of the dataset.

 Which Smart City applicants did you remove? What issues did you see with the documents?
 Removed city: OH Toledo, Issue: Text length (0) is below the threshold (20)
Removed city: CA Moreno Valley, Issue: Text length (0) is below the threshold (20)
Removed city: TX Lubbock, Issue: Text length (0) is below the threshold (20)
Removed city: NV Reno, Issue: Text length (0) is below the threshold (20)
Removed city: FL Tallahassee, Issue: Text length (0) is below the threshold (20)

The text extraction process might have failed for these specific documents.

Explain what additional text processing methods you used and why.
    preprocess_text() function takes in the raw text and a list of extra stopwords as arguments.
    The text is converted to lowercase to maintain uniformity and to match words in the stopwords list.
    The text is tokenized into words using the nltk.word_tokenize() function.
    Only alphabetic words are retained, removing any numbers or special characters.
    Stopwords, including the extra stopwords provided, are removed from the list of words.
    Word lemmatization is performed using the WordNetLemmatizer() class from NLTK, which reduces words to their base form (lemma), improving the clustering process by grouping similar words together.

Did you identify any potientally problematic words?
["city", "state", "smart", "page"]. These words are considered potentially problematic because they are common words that may appear frequently in the documents but do not provide any valuable information for clustering or topic modeling. Removing these words from the text helps to focus on more meaningful words and n-grams that better represent the content of the documents.

## Experimenting with Clustering Models

Now, you'll start to explore models to find the optimal clustering model. In this section, you'll explore [K-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), [Hierarchical](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html), and [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN) clustering algorithms.
Create these algorithms with k_clusters for K-means and Hierarchical.
For each cell in the table provide the [Silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score), [Calinski and Harabasz score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html#sklearn.metrics.calinski_harabasz_score), and [Davies-Bouldin score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html#sklearn.metrics.davies_bouldin_score).

In each cell, create an array to store the values.
|Algorithm| k = 9 | k = 18| k = 36 | Optimal k| 
|--|--|--|--|--|
|K-means|--|--|--|--|
|Hierarchical |--|--|--|--|
|DBSCAN | X | X | X | -- |






 Optimality 
You will need to find the optimal k for K-means and Hierarchical algorithms.
Find the optimality for k in the range 2 to 50.
Provide the code used to generate the optimal k and provide justification for your approach.

evaluating three clustering models (K-means, Hierarchical, and DBSCAN) using the preprocessed text from the Smart City Challenge Applicants dataset. The code can be broken down into the following steps:

    Prepare the data: The preprocessed text is transformed into a term frequency-inverse document frequency (TF-IDF) matrix using the TfidfVectorizer class. This matrix is then converted to a dense NumPy array (X_dense) to be used for some clustering evaluation metrics.

    Initialize variables: Variables to store the range of k-values, results for each clustering model, and the optimal k-values for K-means and Hierarchical clustering (based on silhouette scores) are initialized.

    Evaluate clustering models: The code iterates through each k-value and performs the following steps for both K-means and Hierarchical clustering models:
        Apply the clustering algorithm to the dataset and obtain cluster labels.
        Calculate evaluation metrics (silhouette score, Calinski-Harabasz index, and Davies-Bouldin index) for the current clustering model and store the results.
        Update the optimal k-value for the current clustering model if the silhouette score is better than the previous best silhouette score.

    DBSCAN clustering: Perform DBSCAN clustering on the dataset and obtain cluster labels. If there are enough clusters (more than one unique label), calculate the evaluation metrics (silhouette score, Calinski-Harabasz index, and Davies-Bouldin index) and store the results. Otherwise, store a message indicating that there are not enough clusters.

    Print the results: Print the results dictionary containing the evaluation metrics for each clustering model and their corresponding k-values (for K-means and Hierarchical models) or the message for DBSCAN.

In summary, the code block provided compares three different clustering algorithms (K-means, Hierarchical, and DBSCAN) using various evaluation metrics to determine their performance on the preprocessed Smart City Challenge Applicants dataset. This analysis can help you choose the best clustering model for further analysis or exploration of the dataset.

a function print_results_table() that takes the results dictionary and optimal_k dictionary as inputs and prints a formatted table of the clustering results. The function performs the following steps:

    Create the table header: The header is initialized with the first column named "Algorithm." Next, the sorted k-values from the "K-means" results are added to the header with the format k = {k}. Finally, the "Optimal k" column is added to the header.

    Prepare the table rows: For each clustering algorithm in the results dictionary, the function prepares a row for the table:
        If the algorithm is "DBSCAN", it creates a row with "DBSCAN" as the first element, followed by "X" for each k-value and "--" for the optimal k (as DBSCAN doesn't have a k parameter).
        If the algorithm is "K-means" or "Hierarchical", it creates a row with the algorithm name as the first element, followed by the silhouette score for each k-value (rounded to 4 decimal places), and the optimal k-value from the optimal_k dictionary.

    Create and print the table: The tabulate() function from the tabulate library is used to create a table with the prepared rows and header. The table is formatted using the "grid" style and printed.

After defining the print_results_table() function, it is called with the results and optimal_k dictionaries as arguments to print the formatted table of clustering results. This table provides an organized and easily interpretable view of the silhouette scores and optimal k-values for each clustering algorithm.


How did you approach finding the optimal k?
    For a range of k values (in this case, from 2 to 50), we performed clustering using K-means and Hierarchical clustering algorithms.

    For each k value, we computed three evaluation metrics: Silhouette score, Calinski-Harabasz index, and Davies-Bouldin index. These metrics help us understand the quality of the clustering results.

        Silhouette score: A higher Silhouette score indicates that the clusters are well-separated and the points within a cluster are close to each other.

        Calinski-Harabasz index: A higher Calinski-Harabasz index suggests that the clusters are dense and well-separated.

        Davies-Bouldin index: A lower Davies-Bouldin index implies that the clusters are well-separated and compact.

    We then looked for the k value that yielded the best results in terms of the evaluation metrics mentioned above. In our approach, we considered the k value that maximized the Silhouette score and Calinski-Harabasz index and minimized the Davies-Bouldin index as the optimal k.
    
What algorithm do you believe is the best? Why?
it seems that both K-means and Hierarchical clustering methods have the same optimal k value of 2. To determine which model is best, we can look at the silhouette scores for k=2 for both algorithms.

K-means Silhouette score for k=2: 0.0277
Hierarchical Silhouette score for k=2: 0.0210

Since the silhouette score ranges from -1 to 1, with higher values indicating better cluster separation and cohesion, the K-means clustering algorithm performs slightly better than the Hierarchical clustering algorithm with a silhouette score of 0.0277 compared to 0.0210. So, in this case, the K-means clustering model is preferable.

K-means clustering model using the optimal k-value obtained in the previous step and adds the resulting cluster labels to the DataFrame. It then prints the DataFrame with the added cluster IDs. The Hierarchical clustering model is commented out and not used.

The steps in the code can be described as follows:

    Fit K-means model with optimal k: The K-means clustering model is re-fit with the optimal k-value obtained in the previous step using the KMeans() class. The resulting cluster labels are obtained using the fit_predict() method and stored in kmeans_optimal_labels.

    Add cluster labels to DataFrame: The df DataFrame is updated to include a new column called "K-means Cluster ID" containing the cluster labels obtained from the K-means clustering model.

    Print DataFrame with cluster IDs: The updated df DataFrame is printed to the console using the print() function. This shows the original DataFrame with an additional column indicating the K-means cluster ID for each row.

Note that the Hierarchical clustering model is commented out and not used in this code block. If you wanted to use the Hierarchical model, you could uncomment the relevant code and follow similar steps to obtain the cluster labels and add them to the DataFrame.

K-means clustering model with n_clusters=2 using the KMeans() class from the sklearn.cluster module. The resulting model is saved to a file called 'model.pkl' using the joblib.dump() function from the joblib library.

The steps in the code can be described as follows:

    Fit K-means model: A K-means clustering model with n_clusters=2 is fit to the X matrix using the KMeans() class. The resulting model is stored in the kmeans variable.

    Save model to file: The joblib.dump() function is used to save the kmeans model to a file called 'model.pkl'. This file can be loaded later to use the model for prediction on new data.

    Load K-means model from file: The joblib.load() function is used to load the K-means model from the 'model.pkl' file into a new variable called loaded_kmeans. This allows the saved model to be used for prediction on new data without having to retrain the model from scratch.
    
Derving Themes and Concepts

Perform Topic Modeling on the cleaned data. Provide the top five words for `TOPIC_NUM = Best_k` as defined in the section above. Feel free to reference [Chapter 6](https://github.com/dipanjanS/text-analytics-with-python/tree/master/New-Second-Edition/Ch06%20-%20Text%20Summarization%20and%20Topic%20Models) for more information on Topic Modeling and Summarization.

performs topic modeling using Latent Dirichlet Allocation (LDA) to extract the most important topics from the dataset. The optimal k-value obtained earlier is used to determine the number of topics to extract.

The steps in the code can be described as follows:

    Extract the optimal k-value: The optimal k-value obtained earlier in the analysis is extracted from the optimal_k dictionary and stored in the optimal_k_value variable.

    Fit the LDA model: The LDA model is fit to the X matrix using the LatentDirichletAllocation() class from the sklearn.decomposition module. The number of components is set to the optimal_k_value, and the random state is set to 0.

    Define function to print top words: A function called print_top_words() is defined to print the top words for each topic extracted by the LDA model. The function takes as input the LDA model, the feature names, and the number of top words to display for each topic.

    Print top words for each topic: The print_top_words() function is called with the LDA model, the feature names obtained from the vectorizer, and the number of top words set to 5. This prints the top 5 words for each topic extracted by the LDA model.
    
Topic #0: saw subway mount hudson gun
Topic #1: saw subway mount hudson gun
Topic #2: saw subway mount hudson gun
Topic #3: data transportation system vehicle transit
Topic #4: pky river mill bronx ngun
Topic #5: saw subway mount hudson gun

Add Topid ID to output file
Add the top two topics for each smart city to the data structure.

the topic distribution for each city using the LDA model fitted earlier. The top two topics for each city are then identified by sorting the topic distribution array along the second axis and taking the last two columns.

The steps in the code can be described as follows:

    Obtain the topic distribution for each city: The lda.transform() method is used to obtain the topic distribution for each city in the X matrix. This gives a matrix of shape (n_cities, n_topics).

    Find the top two topics for each city: The np.argsort() function is used to sort the topic distribution array along the second axis (i.e., by topic) and take the last two columns, which correspond to the top two topics for each city.

    Combine the top two topics into a tuple: The top two topics for each city are combined into a tuple using the zip() function and stored in the combined_topics variable.

    Add the combined top two topics to the DataFrame: The combined top two topics are added to the DataFrame as a new column called "Top Topics" using the df["Top Topics"] = combined_topics statement.

    Print the DataFrame with combined top two topics: The resulting DataFrame with the city names, raw text, clean text, K-means cluster ID, and top two topics is printed using the print(df) statement.

exports the final DataFrame as a tab-separated values (TSV) file called "smartcity_eda.tsv" using the to_csv() method of the DataFrame object. The TSV file format is specified using the sep='\t' parameter.

This line of code allows the final DataFrame to be easily shared and used in other applications that can read TSV files, such as Microsoft Excel or Python Pandas.

---------Thank you ---------
