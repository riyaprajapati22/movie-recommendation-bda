# Movie Recommendation System (PySpark ALS + MovieLens)

Mini project: large-data application - Recommendation System.

## Pipeline
1. `download_data.py` - fetch MovieLens dataset (standard public dataset)
2. `analyze.py` - Spark SQL exploratory analysis (ratings, genres, trends)
3. `train.py` - ALS collaborative filtering, RMSE evaluation, top-10 recs per user
4. `app.py` - Streamlit UI (recommend for a user / find similar movies)

## Setup
Needs Python 3.9+ and Java 8/11/17.
```
pip install -r requirements.txt
```

## Run
```
python download_data.py            # add --size 25m for 25M ratings (bigger data)
python analyze.py
python train.py
streamlit run app.py
```

## Scaling note (for viva)
Spark partitions the ratings across cores/nodes; ALS alternates solving user and
item factor matrices in parallel. Switch `master("local[*]")` to a YARN/Spark
cluster and point to HDFS paths to run on a real cluster with no code change
to the algorithm.
# nids-ml
# nids-ml
