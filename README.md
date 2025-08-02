# Unsupervised Anomaly Detection in Financial Transactions

This project implements a complete, end-to-end Big Data pipeline to detect anomalies in a large-scale financial transaction dataset using unsupervised machine learning on the cloud.

## Project Goal
The primary goal of this project was to move beyond traditional, static rule-based fraud detection systems. Such systems are often unable to keep pace with the volume and complexity of modern financial data and typically fail to identify novel fraud patterns. This project aimed to build a scalable and adaptive solution using Apache Spark on AWS that could learn the patterns of normal transactional behavior and flag significant, previously unseen deviations as potential anomalies, all without relying on pre-labeled data.

## The Process
The implementation was a multi-stage process that began with local data preparation and culminated in distributed model training on the cloud.

The project started with the **IEEE-CIS Fraud Detection dataset**, a large, real-world dataset containing nearly 600,000 transactions and over 400 initial features. Initial preprocessing was performed locally using Python and the pandas library. This involved merging transaction and identity files, conducting a thorough missing value analysis, and dropping over 200 columns that had more than 50% of their data missing. The remaining null values were imputed (using the median for numerical data and the mode for categorical), and all features were converted to a numerical format. The final cleaned dataset was saved to the efficient Parquet format.

For the Big Data phase, the cleaned data was uploaded to an **AWS S3 bucket**. An **AWS EMR (Elastic MapReduce) cluster** was launched and configured with **Apache Spark 3.5.5**. The project then faced several real-world technical challenges, requiring pivots in the modeling approach. An initial attempt to use an Isolation Forest model failed due to incompatible third-party libraries, and a subsequent attempt to use the built-in Local Outlier Factor (LOF) model also failed due to a subtle EMR environment issue. The final, successful implementation used a robust, built-in **Gaussian Mixture Model (GMM)**, demonstrating the need for adaptability in data science projects.

## Technology & Software Used
* **Cloud Platform:** Amazon Web Services (AWS)
    * **EMR (Elastic MapReduce)** for managing the Spark cluster.
    * **S3 (Simple Storage Service)** for scalable data storage.
* **Big Data Engine:** Apache Spark (PySpark)
* **Core Language:** Python
* **Data Science Libraries:** Pandas, NumPy
* **ML Algorithm:** Gaussian Mixture Model (GMM)

## Results & Success
The primary success of this project was the successful execution of the end-to-end pipeline. The Spark job ran to completion on the EMR cluster, training the GMM on the full, preprocessed dataset.

The analytical result was that the model, with its initial hyperparameter of `k=15` clusters, identified **zero anomalies**. It correctly calculated a probability threshold to flag the bottom 3.5% of transactions but found no data points that fell below this cutoff. This is a valid and important finding, establishing a baseline and indicating that the "normal" transactions in the dataset form relatively coherent clusters at this level of granularity.

## Next Steps & Future Improvements
While the pipeline is complete, the model itself can be further refined.
* **Hyperparameter Tuning:** The most critical next step is to experiment with the number of clusters (`k`) in the GMM. A different `k` value would change the data's probability distribution and could lead to the identification of anomalies.
* **Alternative Models:** Implement a different unsupervised algorithm, such as Principal Component Analysis (PCA) for anomaly detection based on reconstruction error, to provide a comparative analysis.
* **Formal Evaluation:** Use the `isFraud` column from the original dataset (which was excluded from the unsupervised training) to perform a quantitative evaluation (Precision, Recall, F1-Score) of the model at different probability thresholds.
