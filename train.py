"""Step 2: train ALS collaborative-filtering model, evaluate, export recommendations."""
import os
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from common import get_spark, load

os.makedirs("output", exist_ok=True)
spark = get_spark("TrainALS")
ratings, movies = load(spark)
ratings = ratings.select("userId", "movieId", "rating")

train, test = ratings.randomSplit([0.8, 0.2], seed=42)

als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
          rank=10, maxIter=10, regParam=0.1,
          coldStartStrategy="drop", nonnegative=True)
model = als.fit(train)

pred = model.transform(test)
rmse = RegressionEvaluator(metricName="rmse", labelCol="rating",
                           predictionCol="prediction").evaluate(pred)
print(f"Test RMSE: {rmse:.4f}")
with open("output/metrics.txt", "w") as f:
    f.write(f"RMSE={rmse:.4f}\n")

# Top-10 recommendations for every user -> flat CSV (easy for the UI to read)
recs = model.recommendForAllUsers(10)
flat = (recs.select("userId", F.explode("recommendations").alias("rec"))
        .select("userId", F.col("rec.movieId").alias("movieId"),
                F.col("rec.rating").alias("score"))
        .join(movies, "movieId")
        .select("userId", "movieId", "title", "genres", F.round("score", 3).alias("score")))
flat.toPandas().to_csv("output/recommendations.csv", index=False)

# Similar movies (item-item) using learned item factors
model.itemFactors.select(F.col("id").alias("movieId"), "features") \
    .toPandas().to_pickle("output/item_factors.pkl")
movies.toPandas().to_csv("output/movies.csv", index=False)

print("Saved recommendations.csv, item_factors.pkl, movies.csv in ./output")
spark.stop()
