import glob, os
from pyspark.sql import SparkSession


def get_spark(name="MovieRecSys"):
    return (SparkSession.builder.appName(name)
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "4g")
            .getOrCreate())


def data_dir():
    """Find the extracted MovieLens folder (ml-latest-small or ml-25m)."""
    for d in ["data/ml-latest-small", "data/ml-25m"]:
        if os.path.exists(os.path.join(d, "ratings.csv")):
            return d
    raise FileNotFoundError("Run `python download_data.py` first.")


def load(spark):
    d = data_dir()
    ratings = spark.read.csv(f"{d}/ratings.csv", header=True, inferSchema=True)
    movies = spark.read.csv(f"{d}/movies.csv", header=True, inferSchema=True)
    return ratings, movies
