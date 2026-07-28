"""
PredictWell - Distributed Preprocessing (Diabetes Dataset)
------------------------------------------------------------
Uses Apache PySpark to perform distributed data cleaning and feature
engineering on the Pima Indians Diabetes dataset. Spark is used here
(instead of Pandas) to parallelize:
    - null / zero-value imputation across partitions
    - feature engineering (BMI category, age bucket, glucose-insulin ratio)
    - train/test split preparation

Output: a cleaned CSV written to data/diabetes_clean.csv which is then
consumed by the scikit-learn training script.
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "outcome"
]

# Columns where 0 is not physiologically valid -> treat as missing
ZERO_AS_NULL_COLS = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def build_spark():
    return (
        SparkSession.builder
        .appName("PredictWell-Diabetes-Preprocessing")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def main():
    start = time.time()
    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.csv("data/diabetes.csv", header=False, inferSchema=True)
    df = df.toDF(*COLUMNS)

    # Cast everything to double for uniform numeric processing
    for c in COLUMNS:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))

    # --- Distributed null handling: replace biologically-impossible 0s with null ---
    for c in ZERO_AS_NULL_COLS:
        df = df.withColumn(c, F.when(F.col(c) == 0, None).otherwise(F.col(c)))

    # --- Distributed imputation using column medians (computed via approxQuantile) ---
    medians = {}
    for c in ZERO_AS_NULL_COLS:
        med = df.approxQuantile(c, [0.5], 0.01)[0]
        medians[c] = med
    df = df.na.fill(medians)

    # --- Feature engineering (parallelized across partitions) ---
    df = df.withColumn(
        "bmi_category",
        F.when(F.col("bmi") < 18.5, 0)
         .when(F.col("bmi") < 25, 1)
         .when(F.col("bmi") < 30, 2)
         .otherwise(3)
    )

    df = df.withColumn(
        "age_bucket",
        F.when(F.col("age") < 30, 0)
         .when(F.col("age") < 45, 1)
         .when(F.col("age") < 60, 2)
         .otherwise(3)
    )

    df = df.withColumn(
        "glucose_insulin_ratio",
        F.round(F.col("glucose") / (F.col("insulin") + F.lit(1.0)), 4)
    )

    # Reorder so outcome (label) is last
    feature_cols = [c for c in df.columns if c != "outcome"] + ["outcome"]
    df = df.select(*feature_cols)

    # Write cleaned single CSV file (coalesce for a simple downstream read)
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv("data/diabetes_clean_spark")
    )

    elapsed = time.time() - start
    print(f"[Spark] Diabetes preprocessing complete in {elapsed:.2f}s. Rows: {df.count()}")
    spark.stop()


if __name__ == "__main__":
    main()
