# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('RModels').getOrCreate()



from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator



df = spark.read.csv('/FileStore/tables/movielens_ratings.csv',inferSchema=True,header=True)
df.show()



  df.describe().show()



train_data,test_data = df.randomSplit([0.8,0.2])



als = ALS(maxIter=5,regParam=0.01,userCol='userId', itemCol='movieId', ratingCol='rating')



model=als.fit(train_data)
predictions = model.transform(test_data)



predictions.show()



 evaluator = RegressionEvaluator(metricName='rmse',labelCol='rating',predictionCol='prediction')



rmse = evaluator.evaluate(predictions)



rmse



singleUser = test_data.filter(test_data['userId']==11).select(['movieId','userId'])
singleUser.show()



recommendations = model.transform(singleUser)
recommendations.orderBy('prediction',ascending=True).show()




