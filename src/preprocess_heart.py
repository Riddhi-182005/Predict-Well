"""
PredictWell - Distributed Preprocessing (Heart Disease Dataset)
------------------------------------------------------------------
Same Spark-based cleaning/feature-engineering approach as the diabetes
pipeline, applied to the UCI Cleveland Heart Disease dataset.
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


def build_spark():
    return (
        SparkSession.builder
        .appName("PredictWell-Heart-Preprocessing")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def main():
    start = time.time()
    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.csv("data/heart.csv", header=True, inferSchema=True)

    numeric_cols = [c for c, t in df.dtypes if t in ("int", "double", "bigint")]
    for c in numeric_cols:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))

    # --- Distributed null handling (drop any fully-null rows, impute rest with median) ---
    df = df.na.drop(how="all")
    medians = {}
    for c in numeric_cols:
        if c == "target":
            continue
        med = df.approxQuantile(c, [0.5], 0.01)[0]
        medians[c] = med
    df = df.na.fill(medians)

    # --- Feature engineering ---
    df = df.withColumn(
        "age_bucket",
        F.when(F.col("age") < 40, 0)
         .when(F.col("age") < 55, 1)
         .when(F.col("age") < 65, 2)
         .otherwise(3)
    )

    df = df.withColumn(
        "chol_risk",
        F.when(F.col("chol") < 200, 0)
         .when(F.col("chol") < 240, 1)
         .otherwise(2)
    )

    df = df.withColumn(
        "max_hr_reserve",
        F.round(F.lit(220) - F.col("age") - F.col("thalach"), 2)
    )

    feature_cols = [c for c in df.columns if c != "target"] + ["target"]
    df = df.select(*feature_cols)

    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv("data/heart_clean_spark")
    )

    elapsed = time.time() - start
    print(f"[Spark] Heart preprocessing complete in {elapsed:.2f}s. Rows: {df.count()}")
    spark.stop()


if __name__ == "__main__":
    main()
