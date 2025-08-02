from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import GaussianMixture

def main():
    spark = SparkSession.builder.appName("AnomalyDetectionGMM").getOrCreate()
    s3_input_path = "s3://finalprojectbucket-1x1/data/train_cleaned.parquet"
    df = spark.read.parquet(s3_input_path)
    print(f"Successfully loaded data from {s3_input_path}")

    feature_cols = [c for c in df.columns if c not in ['isFraud', 'TransactionID']]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_assembled = assembler.transform(df)

    gmm = GaussianMixture(featuresCol='features', k=15, seed=1)
    model = gmm.fit(df_assembled)

    predictions = model.transform(df_assembled)
    max_prob_udf = udf(lambda v: float(v.max()), FloatType())
    predictions_with_prob = predictions.withColumn(
        "max_prob", 
        max_prob_udf(predictions.probability)
    )
    
    anomaly_threshold = predictions_with_prob.approxQuantile("max_prob", [0.035], 0.01)[0]
    print(f"Calculated anomaly probability threshold: {anomaly_threshold}")

    anomalies = predictions_with_prob.filter(predictions_with_prob.max_prob < anomaly_threshold)
    
    print("Total count of predicted anomalies:")
    print(anomalies.count())

    anomalies.select("TransactionID", "TransactionAmt", "max_prob").show()

    spark.stop()
    print("SparkSession stopped.")

if __name__ == "__main__":
    main()