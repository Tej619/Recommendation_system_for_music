# -*- coding: utf-8 -*-
"""Recommendation_system.ipynb

Setting up the environment
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install pyspark
!pip install gdown

"""Download Java"""

!apt-get install openjdk-17-jdk -y
!java -version

"""Setting path for java"""

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

"""Downloading Dataset from Kaggle."""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("devdope/900k-spotify")

print("Path to dataset files:", path)

"""# Cleaning Data


"""

import pandas as pd
import numpy as np
import os

print(path)
print(os.listdir(path))

file_path = os.path.join(path, 'spotify_dataset.csv') # get dataset from path

df = pd.read_csv(file_path)

df.head()

"""Checking summary statistics"""

df.describe()

"""Dataset of needed columns"""

# keep only columns needed for recommendation system

cols_needed = ["Genre", "song", "Artist(s)", "emotion", "Release Date", "Tempo"]

df = df[cols_needed].copy()

"""String cleanup"""

# basic cleaning of text columns

text_cols = ["Genre", "song", "Artist(s)", "emotion", "Release Date"]

# strip leading and trailing whitespace and make empty stings NaN

for col in text_cols:
  df[col] = (df[col]
             .astype(str) # ensure string
             .str.strip() # remove extra spaces
             .replace(r'^\s*$', np.nan, regex=True)) # change blank to NaN

"""Droppng missing values from needed fields"""

# drop rows missing values in key identity fields

key_required_fields = ["Genre", "song", "Artist(s)"]

df.dropna(subset=key_required_fields, inplace=True)

# drop rows missing values in supporting fields

df.dropna(subset=["emotion", "Release Date", "Tempo"], inplace=True)

"""Parse and clean the Release Date field"""

# convert Release Date into datetime data type
df["Release Date"] = pd.to_datetime(df["Release Date"], errors = "coerce")

# extract year for later use (e.g., old and recent songs)
df["release_year"] = df["Release Date"].dt.year

# Drop rows with invalid dates if they exist
df = df.dropna(subset=["Release Date", "release_year"])

# checking for missing values
df.isnull().sum()

"""Clean and Validate Tempo"""

# convert Tempo to numeric data type
df["Tempo"] = pd.to_numeric(df["Tempo"], errors="coerce")

# drop rows where tempo is missing or clearly invalid
df = df.dropna(subset=["Tempo"])

"""Check Tempo for outliers"""

# visualizing ditribution of Tempo values to spot outliers

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df["Tempo"], bins=50, edgecolor="black")
plt.title("Tempo Distribution")
plt.xlabel("Tempo (BPM)")
plt.ylabel("Count")
plt.show()

"""Dig deeper by looking at quartiles"""

# calculate the quantiles in inter-quartile range
q1 = df["Tempo"].quantile(0.25)
q3 = df["Tempo"].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

outliers = df[(df["Tempo"] < lower_bound) | (df["Tempo"] > upper_bound)]
print("Outlier count:", len(outliers))

# outlier view
outliers.head()

# filter out extreme outliers (e.g., < 32 or > 205 BPM)
df = df[(df["Tempo"] >= 32) & (df["Tempo"] <= 205)]

"""Normalize text formats"""

# normalize Genre and emotion to lowercase
df["Genre"] = df["Genre"].str.lower()
df["emotion"] = df["emotion"].str.lower()

# renaming columns for easier use
df = df.rename(columns={
    "Genre": "genre",
    "song": "song_name",
    "Artist(s)": "artist_name",
    "Tempo" : "tempo",
    "Release Date" : "release_date"})

# cleaning numerice value for release_year
df["release_year"] = df["release_year"].astype("Int64")

df.head()

df.shape

"""Remove duplicate"""

# drop all duplicae rows
df = df.drop_duplicates()

# drop duplicate songs by artists
df = df.drop_duplicates(subset=["artist_name", "song_name"])

df.shape

"""Save clean dataset"""

df.to_csv("spotify_clean_dataset.csv", index = False)

print("Clean dataset size:", df.shape)

"""Feature Engineering

"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode, trim, collect_set, array_join
from pyspark.sql import functions as F

# It appears there's an issue with the Java environment setup.
# The previous cell trying to install Java 11 failed, but Java 17 is installed.
# The JAVA_HOME environment variable was set to Java 11, which likely caused this error.
# Before running this cell, you need to fix the JAVA_HOME environment variable
# to point to a valid Java installation (e.g., Java 17).
# The fix is provided in cell `6cdIMFdXCS44`.

spark = SparkSession.builder \
    .appName("GenreFeatureEngineering") \
    .getOrCreate()

"""Loading the data"""

df = spark.read.csv("spotify_clean_dataset.csv", header=True, inferSchema=True)

df.show(5, truncate=False)

"""Split genre into singular categories"""

df = df.withColumn("tempo", F.col("tempo").cast("int"))

df_genre_split = df.withColumn("genre", F.split(F.col("genre"), ",")) \
                   .withColumn("genre", F.explode(F.col("genre")))

df_genre_split.show(10, truncate=False)

# df_genre_split.write.mode("overwrite").csv("genre_split.csv", header=True)

"""Normalize the numeric values for tempo"""

# Compute min and max of tempo


tempo_stats = df_genre_split.agg(
    F.min("tempo").alias("min_tempo"),
    F.max("tempo").alias("max_tempo")
).collect()[0]

min_tempo = tempo_stats['min_tempo']
max_tempo = tempo_stats['max_tempo']

# Apply min-max normalization
df_normalized = df_genre_split.withColumn(
    "tempo_normalized",
    (F.col("tempo") - min_tempo) / (max_tempo - min_tempo)
)

df_normalized.show(10, truncate=False)

"""Saved file for both feature engineering and Normalized tempo"""

df_normalized.write.mode("overwrite").csv("featured_eng_normalized_tempo_songs.csv", header=True)

"""Load Data back into Spark"""

df = spark.read.csv("featured_eng_normalized_tempo_songs.csv", header=True, inferSchema=True)
#df.show(5)

"""One-Hot Encode Genre and Emotion"""

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
import pyspark.sql.functions as F

# I came back and added this because Spark Failed with Nulls
df = df.dropna(subset=["genre", "emotion", "tempo_normalized"])

#Prepare for the one hot encoding
#Do this by converting columns from text into numerical indices
genre_indexer = StringIndexer(
    inputCol="genre",
    outputCol="genre_index",
    handleInvalid="keep"
)

emotion_indexer = StringIndexer(
    inputCol="emotion",
    outputCol="emotion_index",
    handleInvalid="keep"
)

#Convert categories into encoded vectors
encoder = OneHotEncoder(
    inputCols=["genre_index", "emotion_index"],
    outputCols=["genre_vec", "emotion_vec"],
    handleInvalid="keep"
)

#combine vectors into a single feature vector
assembler = VectorAssembler(
    inputCols=["genre_vec", "emotion_vec", "tempo_normalized"],
    outputCol="features",
    handleInvalid="keep"
)

#single transformation pipeline
pipeline = Pipeline(stages=[genre_indexer, emotion_indexer, encoder, assembler])

model = pipeline.fit(df)
df_features = model.transform(df)

#VALIDATE FIRST FEW ROWS
#df_features.select(
#    "song_name", "genre", "emotion", "tempo_normalized", "features"
#).show(5, truncate=False)

"""Cosine Similarity"""

import pyspark.sql.functions as F
from pyspark.ml.linalg import DenseVector, SparseVector
import numpy as np

#cosine sim between two vectors
def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

#register function as spark UDF
#Can then be tested on Spark Dataframe
cosine_udf = F.udf(cosine_similarity)

"""Get Top-K Similar Songs (Give Recomendations)"""

def top_k_similar(song_title, k=10):

    # Grab the vector for the requested song
    target = df_features.filter(F.col("song_name") == song_title).select("features").first()

    # Convert Vector to Python list
    target_vec = target["features"].toArray().tolist()

    # Compute similarity with all other songs
    scored = df_features.withColumn(
        "similarity",
        cosine_udf(F.col("features"), F.lit(target_vec))
    )

    # Exclude the song itself and sort by similarity
    result = scored.filter(F.col("song_name") != song_title) \
                   .orderBy(F.col("similarity").desc()) \
                   .select("song_name", "artist_name", "genre", "emotion", "tempo_normalized", "similarity") \
                   .limit(k)

    return result

"""Test Recomendation and Save Results"""

#Will use Taylor Swifts Love story and get the top 10 recomended

results = top_k_similar("Love Story", 10)
results.show(truncate=False)

#output to file
results.write.mode("overwrite").csv("recommendations_love_story.csv", header=True)

# precision score evaluation

def precision_at_k(song_title, results_df):
  # get target song metadata
  meta = df_features.filter(F.col("song_name") == song_title)\
    .select("genre", "emotion")\
    .first()
  target_genre = meta["genre"]
  target_emotion = meta["emotion"]

  # count the top K share in both fields
  relevant = results_df.filter(
      (F.col("genre") == target_genre) &
      (F.col("emotion") == target_emotion)
  ).count()

  # get precision
  K = results_df.count()
  precision = relevant / K
  return precision

# test
precision = precision_at_k("Love Story", results)
print("precision at top 10: ", precision)