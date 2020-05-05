# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ECommerceCustomers').getOrCreate()



from pyspark.ml.regression import LinearRegression



data = spark.read.csv('/FileStore/tables/Ecommerce_Customers.csv',inferSchema=True, header=True)
data.show()



data.printSchema()



from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler



data.columns



assembler = VectorAssembler(inputCols=['Avg Session Length', 'Time on App', 'Time on Website','Length of Membership'],
                           outputCol ='AmountSpent')



output = assembler.transform(data)



output.printSchema()



final_data = output.select('AmountSpent', 'Yearly Amount Spent')
final_data.show()



train_data,test_data = final_data.randomSplit([0.7,0.3])



train_data.describe().show()



test_data.describe().show()



lr = LinearRegression(labelCol='Yearly Amount Spent', featuresCol='AmountSpent')



lr_model = lr.fit(train_data)



test_results = lr_model.evaluate(test_data)



test_results.residuals.show()



test_results.rootMeanSquaredError



test_results.r2



final_data.describe().show()



unlabeled_data =test_data.select('AmountSpent')
unlabeled_data.show()



predictions = lr_model.transform(unlabeled_data)
predictions.show()




