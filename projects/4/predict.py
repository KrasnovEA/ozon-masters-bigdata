from pyspark.sql import SparkSession
import os
import sys
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import Pipeline, PipelineModel

model = PipelineModel.load(str(sys.argv[1]))
df_schema = StructType([
    StructField("overall", FloatType()),
    StructField("vote", StringType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", FloatType()),
])
df = spark.read.json(str(sys.argv[2]), schema = df_schema)
df2 = df.select('overall', 'reviewText')
predictions = model.transform(df2)

predictions.write().overwrite().save(str(sys.argv[3]))
