# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('CrewMembers').getOrCreate()



data = spark.read.csv('/FileStore/tables/cruise_ship_info.csv',header=True,inferSchema=True)
data.show()



data.printSchema()



data.describe().show()



data.groupBy('Cruise_line').count().show()



from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Cruise_line", outputCol="CruiseLine")
indexed = indexer.fit(data).transform(data)
indexed.show()



from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler



indexed.columns



assembler = VectorAssembler(inputCols=['Age',
 'Tonnage',
 'passengers',
 'length',
 'cabins',
 'passenger_density',
 'CruiseLine'], outputCol= 'features')



output = assembler.transform(indexed)



output.select('features','crew').show()



final_data = output.select('features', 'crew')



train_data,test_data = final_data.randomSplit([0.7,0.3])



train_data.describe().show()



test_data.describe().show()



from pyspark.ml.regression import LinearRegression



lr=LinearRegression(labelCol='crew')



model = lr.fit(train_data)



results = model.evaluate(test_data)



results.rootMeanSquaredError



results.meanAbsoluteError



results.r2



results.r2adj



from pyspark.sql.functions import corr



data.select(corr('crew','passengers')).show()




