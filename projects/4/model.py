from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline


tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
hasher = HashingTF(numFeatures=5000, binary=True, inputCol=tokenizer.getOutputCol(), outputCol="word_vector")
lr = LinearRegression(featuresCol=hasher.getOutputCol(), labelCol="overall", maxIter=50)

pipeline = Pipeline(stages=[
    tokenizer,
    hasher,
    lr
])
