"""Step 1: exploratory big-data analysis with Spark SQL. Saves CSVs + charts to ./output"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from common import get_spark, load

os.makedirs("output", exist_ok=True)
spark = get_spark("Analyze")
ratings, movies = load(spark)
ratings.createOrReplaceTempView("ratings")
movies.createOrReplaceTempView("movies")

print("Ratings:", ratings.count(), "| Users:", ratings.select("userId").distinct().count(),
      "| Movies:", ratings.select("movieId").distinct().count())

# 1. Rating distribution
dist = spark.sql("SELECT rating, COUNT(*) AS n FROM ratings GROUP BY rating ORDER BY rating").toPandas()
dist.plot(x="rating", y="n", kind="bar", legend=False, title="Rating distribution")
plt.tight_layout(); plt.savefig("output/rating_distribution.png"); plt.close()

# 2. Top movies (at least 50 ratings)
top = spark.sql("""
    SELECT m.title, ROUND(AVG(r.rating),2) AS avg_rating, COUNT(*) AS n_ratings
    FROM ratings r JOIN movies m ON r.movieId = m.movieId
    GROUP BY m.title HAVING COUNT(*) >= 50
    ORDER BY avg_rating DESC, n_ratings DESC LIMIT 15""").toPandas()
top.to_csv("output/top_movies.csv", index=False)
print(top.to_string(index=False))

# 3. Ratings per genre
genres = (movies.withColumn("genre", F.explode(F.split("genres", "\\|")))
          .join(ratings, "movieId")
          .groupBy("genre").agg(F.count("*").alias("n"), F.round(F.avg("rating"), 2).alias("avg_rating"))
          .orderBy(F.desc("n")).toPandas())
genres.to_csv("output/genres.csv", index=False)
genres.plot(x="genre", y="n", kind="bar", legend=False, title="Ratings per genre")
plt.tight_layout(); plt.savefig("output/genres.png"); plt.close()

# 4. Ratings over time (per year)
yearly = (ratings.withColumn("year", F.year(F.from_unixtime("timestamp")))
          .groupBy("year").count().orderBy("year").toPandas())
yearly.plot(x="year", y="count", legend=False, title="Ratings per year")
plt.tight_layout(); plt.savefig("output/ratings_per_year.png"); plt.close()

print("Analysis saved in ./output")
spark.stop()
