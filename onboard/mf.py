import sys
import math
import numpy as np 

from pyspark.sql import SparkSession, Row
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.recommendation import ALS


# expected percentile ranking
def EPR(predictions, userCol='userId', itemCol='itemId', ratingCol='score'):
    predictions.createOrReplaceTempView('predictions')
    
    # sum of all items
    denominator = predictions.groupBy().sum(ratingCol).collect()[0][0]
    
    # calculate rankings by users
    spark.sql("SELECT " + userCol + " , " + ratingCol + " , PERCENT_RANK() OVER (PARTITION BY " + userCol + " \
              ORDER BY prediction DESC) AS rank FROM predictions").createOrReplaceTempView("rankings")
    
    numerator = spark.sql('SELECT SUM(' + ratingCol + ' * rank) FROM rankings').collect()[0][0]
    
    performance = numerator / denominator
    return performance


if __name__ =='__main__':
	spark = SparkSession\
		.builder\
        .appName("ALSExample")\
        .getOrCreate()

	lines = spark.read.text('../matrix.txt').rdd
	parts = lines.map(lambda row: row.value.split("\t"))

	scoresRDD = parts.map(lambda p: Row(userId=int(p[0]), itemId=int(p[1]), score=float(p[-1])))
	scores = spark.createDataFrame(scoresRDD).select('userId', 'itemId', 'score')

	training, test = scores.randomSplit([0.8, 0.2])

	# use ALS to build recommender
	# set cold start strategy to 'drop'
	# als = ALS(rank=20, maxIter=10, regParam=0.01, 
	# 		  userCol='userId', itemCol='itemId', ratingCol='score', 
	# 		  nonnegative=True, implicitPrefs = True, coldStartStrategy='drop')
	# model = als.fit(training)

	# testPredic = model.transform(test).fillna(0)
	# testPredicData = testPredic.select('userId','itemId','score','label',testPredic.prediction.cast("Double"))
	
	# evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
	# auc = evaluator.evaluate(testPredicData,{evaluator.metricName: 'areaUnderROC'})
	# print(auc)

	# find fitting parameters
	ranks = [10, 20, 50, 100, 150]
	maxIters = [20, 50, 100, 200]
	regParams = [0.05]#[0.05, 0.1, 0.15]
	alphas = [1]#[1, 10, 15]

	best_r = best_i = best_rp = best_a = 0
	best_performance = float('inf')

	for r in ranks:
	    for i in maxIters:
	        for rp in regParams:
	            for a in alphas:
	                als = ALS(rank=r, maxIter=i, regParam=rp, alpha=a,
	                          userCol='userId', itemCol='itemId', ratingCol='score', 
	                          implicitPrefs=True, nonnegative=True, coldStartStrategy='drop')
	                model = als.fit(training)
	                predictions = model.transform(test).fillna(0)
	                performance = EPR(predictions)
	                
	                print("Model Parameters: Rank={}, maxIter={}, regParam={}, alpha={}".format(r, i, rp, a))
	                print("EPR Error: ", performance)
	                
	                if performance < best_performance:
	                    best_r = r
	                    best_i = i
	                    best_rp = rp
	                    best_a = a
	                    best_performance = performance

	print("Best Model Parameters: Rank={}, maxIter={}, regParam={}, alpha={}".format(best_r, best_i, best_rp, best_a))
	print("Best EPR Performance: {}".format(best_performance)) 

	spark.stop()



