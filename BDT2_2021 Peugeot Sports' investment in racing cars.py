# Databricks notebook source
# Name: Zheng Bin
# Academic year: 2021 2ndSession
# Course Name: BDT2

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md #Set the path

# COMMAND ----------

#Set the path below:
circuits_filepath = '/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/circuits.csv'
constructorResults_filepath = '/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/constructorResults.csv'
constructors_filepath = '/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/constructors.csv'
drivers_filepath = '/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/drivers.csv'
pitStops_filepath = '/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/pitStops.csv'
races_filepath = '/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/races.csv'
results_filepath = '/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/results.csv'
status_filepath = '/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/status.csv'
lapTimes_filepath = '/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/lapTimes.csv'

# COMMAND ----------

# MAGIC %md #Reading data

# COMMAND ----------

# Read in data

circuits = spark.read.format("csv").option("header","true").option("inferSchema","true").load(circuits_filepath)

constructorResults = spark.read.format("csv").option("header","true").option("inferSchema","true").load(constructorResults_filepath)

constructors = spark.read.format("csv").option("header","true").option("inferSchema","true").load(constructors_filepath)

drivers = spark.read.format("csv").option("header","true").option("inferSchema","true").load(drivers_filepath)

pitStops = spark.read.format("csv").option("header","true").option("inferSchema","true").load(pitStops_filepath)

races = spark.read.format("csv").option("header","true").option("inferSchema","true").load(races_filepath)

results = spark.read.format("csv").option("header","true").option("inferSchema","true").load(results_filepath)

status = spark.read.format("csv").option("header","true").option("inferSchema","true").load(status_filepath)

lapTimes = spark.read.format("csv").option("header","true").option("inferSchema","true").load(lapTimes_filepath)

print('circuits')
circuits.display()
circuits.printSchema()

print('constructorResults')
constructorResults.display()
constructorResults.printSchema()

print('constructors')
constructors.display()
constructors.printSchema()

print('drivers')
drivers.display()
drivers.printSchema()

print('pitStops')
pitStops.display()
pitStops.printSchema()

print('races')
races.display()
races.printSchema()

print('results')
results.display()
results.printSchema()

print('status')
status.display()
status.printSchema()

print('lapTimes')
lapTimes.display()
lapTimes.printSchema()

# COMMAND ----------

# MAGIC %md # Q1-Models 1 (ML) : Predict whether a driver will finish the race 

# COMMAND ----------

# Build a machine learning model that can predict whether a driver will finish the race or not. You can use the column statusld in the tables results and status to get more information on whether a racer finished the race or not. Use at least two algorithms in your modeling phase. What are your predictions for which drivers will and which drivers won't finish the race for the Abu Dhabi Grand Prix for 2018? Give the probabilities per driver participating in the race

# COMMAND ----------

# MAGIC %md ##Data Preprocessing

# COMMAND ----------

# MAGIC %md ### Merge 3 data 

# COMMAND ----------

# join 3 data (results_status_races)
results_status = results.join(status,'statusId',"leftouter")
# transform all finished in state to 1 otherwise 0
results_status=results_status.withColumn('label',when(col('status') == "Finished",lit(1)).otherwise(0)).drop('status')
results_status_races=races.join(results_status,'raceId','leftouter')
results_status_races.display()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ###filter the data with inner join

# COMMAND ----------

#due to the 'Abu Dhabi Grand Prix' was included since 2009, hence only keep the data from 2009 onwards.
results_status_races.where(col('name')=='Abu Dhabi Grand Prix').select(col('year')).distinct().orderBy('year').display()
a=results_status_races.where(col('name')=='Abu Dhabi Grand Prix').select(col('year')).distinct()
results_status_races1=results_status_races.join(a,'year',"inner")

# COMMAND ----------

# # For choose all the data instead of being filtered
# results_status_races1=results_status_races

# COMMAND ----------

# MAGIC %md ###checking the features

# COMMAND ----------

#check the features
results_status_races1.where(col('name')=='Abu Dhabi Grand Prix').select('circuitId','name').display()
results_status_races1.orderBy('year','round').display()
results_status_races1.dtypes

# COMMAND ----------

# MAGIC %md ### dropping data & fill NA & convert the data type

# COMMAND ----------

#Dropping Unnecessary Columns - Dropping features that wouldn't contribute to the model's performance, and also dropping complex features.
rsl_dropna70=results_status_races1.drop('position','positionText','time','fastestLapTime','name','url','time','statusId')
rsl_dropna70.display()
rsl_dropna70.dtypes

#Replace the unknown values with null values for easier handling.
rsl_na=rsl_dropna70.na.fill(-1, subset=["milliseconds","fastestLap","rank",'fastestLapSpeed'])

# recheck missing value
rsl_na.select([(count(when(col(c).isNull() | isnan(c)|(col(c)=='NA') |( col(c)=='R'), c))/rsl_na.count()).alias(c) for c in rsl_na.columns]).display()
rsl_na.display()

#Replace all 'NA' in a column with -1
rsl_na1=rsl_na.withColumn("na_" + 'milliseconds' ,when(col('milliseconds') == "NA",lit(-1)).otherwise(col('milliseconds'))).drop('milliseconds')
rsl_na2=rsl_na1.withColumn("na_" + 'fastestLap' ,when(col('fastestLap') == "NA",lit(-1)).otherwise(col('fastestLap'))).drop('fastestLap')
rsl_na3=rsl_na2.withColumn("na_" + 'rank' ,when(col('rank') == "NA",lit(-1)).otherwise(col('rank'))).drop('rank')
rsl_na3.display()
rsl_na3.dtypes

# convert strings to double
rsl_type1=rsl_na3.withColumn("na_milliseconds",col("na_milliseconds").cast("double"))
rsl_type2=rsl_type1.withColumn("na_fastestLap",col("na_fastestLap").cast("double"))
rsl_type3=rsl_type2.withColumn("na_rank",col("na_rank").cast("double"))
rsl_type3.display()
rsl_type3.dtypes

# COMMAND ----------

rsl_type3.where(col('year')==2018).orderBy('year','round').display()
rsl_type3.where(col('circuitId')==24).orderBy('year','round').display()
rsl_type3.display()

# COMMAND ----------

# MAGIC %md ## building new features

# COMMAND ----------

# MAGIC %md ### 1. Build an age column of all drivers at the time of the race

# COMMAND ----------

# join the drivers data for getting the birthdate
b=drivers.select('driverId','dob','nationality')
rsld=rsl_type3.join(b,'driverId','leftouter')
rsld.display()

# COMMAND ----------

#Calculate the recency of each user's posting behavior from 1 Jan 2015
rsld1=rsld.withColumn("date",to_date(col("date")))
rsld2=rsld1.withColumn("dob",to_date(col("dob"),'d/M/yyyy'))
rsld3=rsld2.withColumn("age",round(datediff(col('date'),col("dob"))/365,0)).drop('dob')
rsld3.display()

# COMMAND ----------

rsld3.orderBy('driverId','year').display()

# COMMAND ----------

# MAGIC %md ### 2. Build a column for each driver's history of total finished

# COMMAND ----------

#Total number of times completed by each driver
acc_sum=results_status_races.selectExpr(
    'driverId','raceId','label',
    "sum(label) over (partition by driverId order by raceId asc) as acc_finished"
 ).select('driverId','raceId','acc_finished')
acc_sum=acc_sum.na.drop("any")
acc_sum.display()

# COMMAND ----------

#join the circuits & acc_sum data
rslc=rsld3.join(circuits,'circuitId','leftouter')
rslc=rslc.join(acc_sum,['driverId','raceId'],'leftouter')
rslc.where(col('year')==2018).display()
rslc.orderBy('driverId','raceId').display()

# COMMAND ----------

# Build a column for each driver's history of total finished
all_acc=rsld3.select('driverId').distinct().join(results_status_races,'driverId','leftouter')
hist_fini=all_acc.groupBy('driverId').agg(sum("label")).withColumnRenamed("sum(label)","acc_finished")
#drops rows for which all columns are na or NaN
hist_fini1=hist_fini.na.drop("all")
hist_fini1.display()

# COMMAND ----------

rslc1=rslc.select('circuitId','driverId','year','date','raceId','round','acc_finished','age','nationality','label')
rslc1.display()

# COMMAND ----------

# Check the number for missing values, which have an extra value in  2017
print(rslc1.where(col('year')==2018).count())
print(rslc1.where(col('driverId').isNull()).count())

# COMMAND ----------

rslc1.where(col('driverId').isNull() | col('driverId').isNull()).display()
# recheck missing value
rslc1.drop('date').select([(count(when(col(c).isNull() | isnan(c)|(col(c)=='NA') |( col(c)=='R'), c))/rslc1.drop('date').count()).alias(c) for c in rslc1.drop('date').columns]).display()
rslc1.dtypes

# COMMAND ----------

# MAGIC %md ##Modelling

# COMMAND ----------

# MAGIC %md ###OneHotEncoder

# COMMAND ----------

rslc1=rslc1.na.fill('unknown', subset=["nationality"])
basetable=rslc1.na.fill(-1, subset=["label"]).drop('resultId')
basetable_final=basetable.withColumn("label",col("label").cast("double"))
basetable_final.display()

# COMMAND ----------

#Create categorical variables for raceId and constructorId
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline

#raceId
raceIdIndxr = StringIndexer().setInputCol("raceId").setOutputCol("raceIdInd")

#circuitId
circuitIdIndxr = StringIndexer().setInputCol("circuitId").setOutputCol("circuitIdInd")

#circuitId
yearIndxr = StringIndexer().setInputCol("year").setOutputCol("yearInd")

#circuitId
roundIndxr = StringIndexer().setInputCol("round").setOutputCol("roundInd")

#circuitId
nationalityIndxr = StringIndexer().setInputCol("nationality").setOutputCol("nationalityInd")


#One-hot encoding
ohee_catv = OneHotEncoder(inputCols=["circuitIdInd","nationalityInd",'raceIdInd'],outputCols=["circuitId_dum","nationality_dum","raceIdInd_dum"])
pipe_catv = Pipeline(stages=[raceIdIndxr, circuitIdIndxr,yearIndxr,roundIndxr,nationalityIndxr, ohee_catv])

basetable_final = pipe_catv.fit(basetable_final).transform(basetable_final)


basetable_final = basetable_final.drop("nationalityInd","circuitIdInd",'raceIdInd')
basetable_final.display()

# COMMAND ----------

# MAGIC %md ### Splitting data

# COMMAND ----------

#drop the na rows
basetable_final1=basetable_final.na.drop('any')
#Create a train and test set with a 70% train, 30% test split
basetable_train, basetable_test = basetable_final1.randomSplit([0.7, 0.3],seed=123)

print(basetable_train.count())
print(basetable_test.count())

# COMMAND ----------

# #drop the na rows
# basetable_final1=basetable_final.na.drop('any')
# #split data by circuitId=24, which is the Abu Dhabi Grand Prix .
# basetable_test=basetable_final1.where(col('circuitId')==24)
# basetable_train=basetable_final1.where(col('circuitId')!=24)
# print(basetable_train.count())
# print(basetable_test.count())

# COMMAND ----------

# Confirming the above change, by checking the delivery or takeout column
basetable_final1.select('label').groupBy('label').count().show()

# COMMAND ----------

#Drop raceId and circuitId from the basetable
basetable_final1=basetable_final1.drop("raceId", "circuitId",'date','nationality')
basetable_train = basetable_train.drop("raceId", "circuitId",'date','nationality')
basetable_test = basetable_test.drop("raceId", "circuitId",'date','nationality')
#Check the file by going to "Data" => "Add Data" => "Create New Table" => "DBFS" => Folder "tmp" => Select the file => Push "Preview Table"

# COMMAND ----------

# MAGIC %md #### Output data in case of the system crash

# COMMAND ----------

#Write data in parquet format
basetable_final.write.format("parquet")\
  .mode("overwrite")\
  .save("/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/basetable_final.parquet")

basetable_final1.write.format("parquet")\
  .mode("overwrite")\
  .save("/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/basetable_final1.parquet")

basetable_train.write.format("parquet")\
  .mode("overwrite")\
  .save("/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/basetable_train.parquet")

basetable_test.write.format("parquet")\
  .mode("overwrite")\
  .save("/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/basetable_test.parquet")

hist_fini1.write.format("parquet")\
  .mode("overwrite")\
  .save("/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/hist_fini1.parquet")

# COMMAND ----------

# read the output data
basetable_final = spark.read.format("parquet").load('/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/basetable_final.parquet')
basetable_final1 = spark.read.format("parquet").load('/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/basetable_final1.parquet')
basetable_train = spark.read.format("parquet").load('/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/basetable_train.parquet')
basetable_test = spark.read.format("parquet").load('/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/basetable_test.parquet')
hist_fini1 = spark.read.format("parquet").load('/FileStore/tables/Data_IndividualProject_BDT2_2021_RETAKE/hist_fini1.parquet')

# COMMAND ----------

#Drop raceId and circuitId from the basetable
basetable_final1=basetable_final1.drop('year','round')
basetable_train = basetable_train.drop('year','round')
basetable_test = basetable_test.drop('year','round')

# COMMAND ----------

# MAGIC %md ###Transform the tables in a table of label, features format

# COMMAND ----------

# Build a classification model to predict finish (1/0) using column "label".
#Transform the tables in a table of label, features format
from pyspark.ml.feature import RFormula

trainBig = RFormula(formula="label ~ . - driverId").fit(basetable_final1).transform(basetable_final1)
train = RFormula(formula="label ~ . - driverId").fit(basetable_train).transform(basetable_train)
test = RFormula(formula="label ~ . - driverId").fit(basetable_test).transform(basetable_test)
print("trainBig nobs: " + str(trainBig.count()))
print("train nobs: " + str(train.count()))
print("test nobs: " + str(test.count()))

# COMMAND ----------

# MAGIC %md ### Build a classification model 

# COMMAND ----------

# Build a classification model to predict finish (1/0) using column "label".

# COMMAND ----------

# MAGIC %md ####1.Logistic Regression model

# COMMAND ----------

#Train a Logistic Regression model
from pyspark.ml.classification import LogisticRegression

#Define the algorithm class
lr = LogisticRegression()

#Fit the model
lrModel = lr.fit(trainBig)

trainBig[['features']].display()

#Notice the size of coefficients array
lrModel.coefficients.toArray().size

# COMMAND ----------

# MAGIC %md #####Hyperparameter tuning

# COMMAND ----------

#Hyperparameter tuning for different hyperparameter values of LR (aka model selection)
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Define pipeline
lr = LogisticRegression()
pipeline = Pipeline().setStages([lr])

#Set param grid
params = ParamGridBuilder()\
  .addGrid(lr.regParam, [0.1, 0.01])\
  .addGrid(lr.maxIter, [50, 100,150])\
  .build()

#Evaluator: uses max(AUC) by default to get the final model
evaluator = BinaryClassificationEvaluator()
#Check params through: evaluator.explainParams()

#Cross-validation of entire pipeline
cv = CrossValidator()\
  .setEstimator(pipeline)\
  .setEstimatorParamMaps(params)\
  .setEvaluator(evaluator)\
  .setNumFolds(5) # Here: 5-fold cross validation

#Run cross-validation, and choose the best set of parameters.
#Spark automatically saves the best model in cvModel.
cvModel = cv.fit(train)
#Get best tuned parameters of pipeline
cvBestPipeline = cvModel.bestModel
cvBestLRModel = cvBestPipeline.stages[-1]._java_obj.parent() #the stages function refers to the stage in the pipelinemodel

print("Best LR model:")
print("** regParam: " + str(cvBestLRModel.getRegParam()))
print("** maxIter: " + str(cvBestLRModel.getMaxIter()))

# COMMAND ----------

# MAGIC %md #####Predict labels of test set using built model

# COMMAND ----------

preds = cvModel.transform(test)\
  .select("prediction", "label")
preds.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Get model performance on test set

# COMMAND ----------

#Get model performance on test set
from pyspark.mllib.evaluation import BinaryClassificationMetrics

out = preds.rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = BinaryClassificationMetrics(out)

print(metrics.areaUnderPR) #area under precision/recall curve
print(metrics.areaUnderROC)#area under Receiver Operating Characteristic curve

# COMMAND ----------

# MAGIC %md
# MAGIC ####  2.Build RandomForest Model

# COMMAND ----------

train.display()

# COMMAND ----------

train.select

# COMMAND ----------

#Exercise: Train a RandomForest model and tune the number of trees for values [150, 300, 500]
from pyspark.ml.classification import RandomForestClassifier

#Define pipeline
rfc = RandomForestClassifier()
rfPipe = Pipeline().setStages([rfc])

#Set param grid
rfParams = ParamGridBuilder()\
  .addGrid(rfc.numTrees, [150, 300, 500])\
  .build()

rfCv = CrossValidator()\
  .setEstimator(rfPipe)\
  .setEstimatorParamMaps(rfParams)\
  .setEvaluator(BinaryClassificationEvaluator())\
  .setNumFolds(5) # Here: 5-fold cross validation

#Run cross-validation, and choose the best set of parameters.
rfcModel = rfCv.fit(train)


# COMMAND ----------

# MAGIC %md ##### Get predictions on the test set

# COMMAND ----------

#Get predictions on the test set
preds = rfcModel.transform(test)
preds.display()

# COMMAND ----------

#Get model accuracy
print("accuracy: " + str(evaluator.evaluate(preds)))

#Get AUC
metrics = BinaryClassificationMetrics(preds.rdd.map(lambda x: (float(x[0]), float(x[1]))))
print("AUC: " + str(metrics.areaUnderROC))

# COMMAND ----------

# MAGIC %md ##### cvModel MulticlassMetrics

# COMMAND ----------

#Get more metrics
from pyspark.mllib.evaluation import MulticlassMetrics

#Cast a DF of predictions to an RDD to access RDD methods of MulticlassMetrics
preds_labels = cvModel.transform(test)\
  .select("prediction", "label")\
  .rdd.map(lambda x: (float(x[0]), float(x[1])))

metrics = MulticlassMetrics(preds_labels)

print("accuracy = %s" % metrics.accuracy)

# COMMAND ----------

# MAGIC %md #####Get more metrics

# COMMAND ----------

#Get more metrics
from pyspark.mllib.evaluation import MulticlassMetrics

labels = preds.rdd.map(lambda lp: lp.label).distinct().collect()
for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics.precision(label)))
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

# COMMAND ----------

# MAGIC %md # Building the predict data wiht Abu Dhabi Grand Prix for 2018

# COMMAND ----------

# MAGIC %md ####all drivers

# COMMAND ----------

# Build a column for each driver's history of total finished
hist_fini=results_status_races.groupBy('driverId').agg(sum("label")).withColumnRenamed("sum(label)","acc_finished")
#drops rows for which all columns are na or NaN
hist_fini1=hist_fini.na.drop("all")
hist_fini1.display()

# COMMAND ----------

hist_fini1.dtypes

# COMMAND ----------

# MAGIC %md #### build the test data

# COMMAND ----------

a=basetable_final.where((col('circuitId')==24) & (col('year')==2018)).drop('driverId',"raceId","circuitId","acc_finished")
hist_fini2=hist_fini1.withColumn("label",lit(-1.0))
# find the age of each driver in 2018
b=drivers.select('driverId','dob')
pred_data=hist_fini2.join(a,'label','leftouter').join(b,'driverId','leftouter')
pred_data=pred_data.withColumn("dob",to_date(col("dob"),'d/M/yyyy'))
pred_data=pred_data.withColumn("age",round(datediff(col('date'),col("dob"))/365,0)).drop('dob','date')
pred_data.display()

# COMMAND ----------

pred_data=pred_data.na.drop('any')
pred_data.display()

# COMMAND ----------

 pred_data=pred_data.drop('year','round','nationality')

# COMMAND ----------

print(test.dtypes)
print(pred_data.dtypes)

# COMMAND ----------

# MAGIC %md ####Transform the tables in a table of label, features format

# COMMAND ----------

test1 = RFormula(formula="label ~ . - driverId").fit(pred_data).transform(pred_data)
test1.display()

# COMMAND ----------

test.display()

# COMMAND ----------

# MAGIC %md ###choose a model

# COMMAND ----------

#Get predictions on the test set of rfcModel
from pyspark.ml.feature import RFormula
preds = rfcModel.transform(test1)
preds.select("driverId", "probability", "prediction").display()

# COMMAND ----------



# COMMAND ----------

preds1 = cvModel.transform(test1)
preds1.select("driverId", "probability", "prediction").display()

# COMMAND ----------

preds.select('prediction').agg(sum('prediction')).show()

# COMMAND ----------

preds1.select('prediction').agg(sum('prediction')).show()
preds.select('prediction').agg(sum('prediction')).show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md #Q2-Models 1 (DL) : Build a Deep Learning model to predict how many pit stops a driver will need per race. 

# COMMAND ----------

# MAGIC %md ##Data Preprocessing

# COMMAND ----------

from pyspark.sql.functions import *
%matplotlib inline

# COMMAND ----------

# MAGIC %md ### Merge 3 data 

# COMMAND ----------

results_status_races.display()

# COMMAND ----------

# MAGIC %md ### dropping data & fill NA & convert the data type

# COMMAND ----------

#Dropping Unnecessary Columns - Dropping features that wouldn't contribute to the model's performance, and also dropping complex features.
rsl_dropna70=results_status_races.drop('position','positionText','time','fastestLapTime','name','url','time','statusId')
rsl_dropna70.display()
rsl_dropna70.dtypes

#Replace the unknown values with null values for easier handling.
rsl_na=rsl_dropna70.na.fill(-1, subset=["milliseconds","fastestLap","rank",'fastestLapSpeed'])

# recheck missing value
rsl_na.select([(count(when(col(c).isNull() | isnan(c)|(col(c)=='NA') |( col(c)=='R'), c))/rsl_na.count()).alias(c) for c in rsl_na.columns]).display()
rsl_na.display()

#Replace all 'NA' in a column with -1
rsl_na1=rsl_na.withColumn("na_" + 'milliseconds' ,when(col('milliseconds') == "NA",lit(-1)).otherwise(col('milliseconds'))).drop('milliseconds')
rsl_na2=rsl_na1.withColumn("na_" + 'fastestLap' ,when(col('fastestLap') == "NA",lit(-1)).otherwise(col('fastestLap'))).drop('fastestLap')
rsl_na3=rsl_na2.withColumn("na_" + 'rank' ,when(col('rank') == "NA",lit(-1)).otherwise(col('rank'))).drop('rank')
rsl_na3.display()
rsl_na3.dtypes

# convert strings to double
rsl_type1=rsl_na3.withColumn("na_milliseconds",col("na_milliseconds").cast("double"))
rsl_type2=rsl_type1.withColumn("na_fastestLap",col("na_fastestLap").cast("double"))
rsl_type3=rsl_type2.withColumn("na_rank",col("na_rank").cast("double"))
rsl_type3.display()
rsl_type3.dtypes

# COMMAND ----------

# MAGIC %md ## building new features

# COMMAND ----------

# MAGIC %md ### 1. Build an age column of all drivers at the time of the race

# COMMAND ----------

# join the drivers data for getting the birthdate
b=drivers.select('driverId','dob','nationality')
rsld=rsl_type3.join(b,'driverId','leftouter')
rsld.display()

# COMMAND ----------

#Calculate the recency of each user's posting behavior from 1 Jan 2015
rsld1=rsld.withColumn("date",to_date(col("date")))
rsld2=rsld1.withColumn("dob",to_date(col("dob"),'d/M/yyyy'))
rsld3=rsld2.withColumn("age",round(datediff(col('date'),col("dob"))/365,0)).drop('dob')
rsld3.display()

# COMMAND ----------

rsld3.orderBy('driverId','year').display()

# COMMAND ----------

# MAGIC %md ### 2. Build a column for each driver's history of total finished

# COMMAND ----------

#Total number of times completed by each driver
acc_sum=results_status_races.selectExpr(
    'driverId','raceId','label',
    "sum(label) over (partition by driverId order by raceId asc) as acc_finished"
 ).select('driverId','raceId','acc_finished')
acc_sum=acc_sum.na.drop("any")
acc_sum.display()

# COMMAND ----------

#join the circuits & acc_sum data
rslc=rsld3.join(circuits,'circuitId','leftouter')
rslc=rslc.join(acc_sum,['driverId','raceId'],'leftouter')
rslc.where(col('year')==2018).display()
rslc.orderBy('driverId','raceId').display()

# COMMAND ----------

# Build a column for each driver's history of total finished
all_acc=rsld3.select('driverId').distinct().join(results_status_races,'driverId','leftouter')
hist_fini=all_acc.groupBy('driverId').agg(sum("label")).withColumnRenamed("sum(label)","acc_finished")
#drops rows for which all columns are na or NaN
hist_fini1=hist_fini.na.drop("all")
hist_fini1.display()

# COMMAND ----------

rslc1=rslc.select('circuitId','driverId','year','date','raceId','round','acc_finished','age','nationality','label')
rslc1.display()

# COMMAND ----------

# Check the number for missing values, which have an extra value in  2017
print(rslc1.where(col('year')==2018).count())
print(rslc1.where(col('driverId').isNull()).count())

# COMMAND ----------

rslc1.where(col('driverId').isNull() | col('driverId').isNull()).display()
# recheck missing value
rslc1.drop('date').select([(count(when(col(c).isNull() | isnan(c)|(col(c)=='NA') |( col(c)=='R'), c))/rslc1.drop('date').count()).alias(c) for c in rslc1.drop('date').columns]).display()
rslc1.dtypes

# COMMAND ----------

# MAGIC %md ### 3. Building a data with number of pit stop of a driver had in each previous race

# COMMAND ----------

#Create a struct to build a data with number of pit stop of a driver had in each previous race
pitStops1=pitStops.withColumn("complex",struct('raceId','driverId'))
npit=pitStops1.groupBy('complex').agg(sum('stop')).select("complex.*",'sum(stop)').withColumnRenamed('sum(stop)',"N_stop")
npit.display()

# COMMAND ----------

rslcs=rslc1.join(npit,['driverId','raceId'],'leftouter').na.fill(0, subset=['N_stop']).na.drop('any')
rslcs.display()

# COMMAND ----------

rslcs.select('label').distinct().show()

# COMMAND ----------

# MAGIC %md #### rechecking missing value

# COMMAND ----------

rslcs.where(col('N_stop')>=0).display()
# recheck missing value
rslcs.drop('date').select([(count(when(col(c).isNull() | isnan(c)|(col(c)=='NA') |( col(c)=='R'), c))/rslcs.drop('date').count()).alias(c) for c in rslcs.drop('date').columns]).display()

# COMMAND ----------

# MAGIC %md ####[basetable_final] drop any row with missing value

# COMMAND ----------

basetable_final=rslcs.na.drop('any')
# recheck missing value
basetable_final.drop('date').select([(count(when(col(c).isNull() | isnan(c)|(col(c)=='NA') |( col(c)=='R'), c))/basetable_final.drop('date').count()).alias(c) for c in basetable_final.drop('date').columns]).display()

# COMMAND ----------

# MAGIC %md ##one-hot encoding of categorical variables

# COMMAND ----------

#Perform one-hot encoding of categorical variables
from pyspark.sql.functions import *
categ = basetable_final.select('nationality').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [when(col('nationality') == cat,1).otherwise(0).alias("nationality_"+str(cat)) for cat in categ]
basetable_transf = basetable_final.select(exprs+basetable_final.columns)

categ = basetable_final.select('raceId').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [when(col('raceId') == cat,1).otherwise(0).alias("raceId_"+str(cat)) for cat in categ]
basetable_transg = basetable_transf.select(exprs+basetable_transf.columns)

categ = basetable_final.select('circuitId').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [when(col('circuitId') == cat,1).otherwise(0).alias("circuitId_"+str(cat)) for cat in categ]
basetable_transfh = basetable_transg.select(exprs+basetable_transf.columns)

categ = basetable_final.select('N_stop').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [when(col('N_stop') == cat,1).otherwise(0).alias("N_stop_"+str(cat)) for cat in categ]
basetable_transf1 = basetable_transfh.select(exprs+basetable_transf.columns)

basetable_transf1 = basetable_transf1.drop("nationality","raceId","circuitId","date","N_stop_0","N_stop")

#rename columns
basetable_transf2=basetable_transf1#.withColumnRenamed("label","if_finish").withColumnRenamed("N_stop","label")
basetable_transf2.display()

# COMMAND ----------

# MAGIC %md #Single-label Multiclass Classification of DL

# COMMAND ----------

# MAGIC %md ####Transform to numpy arrays

# COMMAND ----------

#Transform to numpy arrays
import numpy as np
x_reg = np.array(basetable_transf2.drop("driverId",'N_stop_6','N_stop_5','N_stop_1','N_stop_10','N_stop_3','N_stop_21','N_stop_15').collect())
y_reg = np.array(basetable_transf2.select('N_stop_6','N_stop_5','N_stop_1','N_stop_10','N_stop_3','N_stop_21','N_stop_15').collect())

# COMMAND ----------

# MAGIC %md ##Split

# COMMAND ----------

#Split in a train, val, and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_reg, y_reg, test_size=0.25, random_state=123)
partial_x_train, x_val, partial_y_train, y_val = train_test_split(x_train, y_train, test_size=0.50, random_state=123)

# COMMAND ----------

y_test

# COMMAND ----------

print(x_train.shape)
print(y_train.shape)
print(partial_x_train.shape)
print(partial_y_train.shape)

# COMMAND ----------

# MAGIC 
# MAGIC %md ## Model: 5 layers 

# COMMAND ----------

#Use the same number of units for the first 4 layers.
#For these first 4 layers, use a number of units that is larger than the number of output classes but smaller than 100 (taking into account the 2^n rule).

#Instead of 16 hidden units to learn 46 classes, use 64 units. 16 units is too restrictive: every layer could serve as a bottleneck
from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu')) 
model.add(layers.Dense(64, activation='relu')) 
model.add(layers.Dense(7, activation='softmax')) #output a probability distribution over the 7 different output classes.
#You end up with a network with a Dense layer of size 7

# COMMAND ----------

# MAGIC %md ###Define an optimizer, loss function, and metric for success

# COMMAND ----------

x_train.shape[1]

# COMMAND ----------

#Define an optimizer, loss function, and metric for success
model.compile(optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['acc'])
#(/0.5)

#Fit the model. Use batch_size=512 and epoch=20 as start values.
history = model.fit(partial_x_train,
  partial_y_train,
  epochs=20,
  batch_size=512,
  validation_data=(x_val, y_val))

# COMMAND ----------

# MAGIC %md ###Create plot

# COMMAND ----------

#Look at results (accuracy)
import matplotlib.pyplot as plt
plt.clf()

#Create values
acc_values = history.history['acc']
val_acc_values = history.history['val_acc']
epochs = range(1, len(acc_values) + 1)

#Create plot
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# COMMAND ----------

#Network seems to overfit after 5 epochs (see graphs)

model.compile(optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['acc'])

model.fit(x_train, y_train, epochs=5, batch_size=512)

results_optim_epoch = model.evaluate(x_test, y_test)

# COMMAND ----------

results_optim_epoch

# COMMAND ----------

# MAGIC %md ##Model: 3 layers

# COMMAND ----------

#Estimate a model with 2 layers of 64 units and 1 output layer.
###############################################################
from tensorflow.keras import models
from tensorflow.keras import layers
model = models.Sequential()

model.add(layers.Dense(64, activation='relu',
  input_shape=(x_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

# COMMAND ----------

#Define an optimizer, loss function, and metric for success
model.compile(optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['acc'])


#Fit the model. Use batch_size=512 and epoch=20 as start values.
history = model.fit(partial_x_train,
  partial_y_train,
  epochs=20,
  batch_size=512,
  validation_data=(x_val, y_val))

# COMMAND ----------

#Look at results (accuracy)
import matplotlib.pyplot as plt
plt.clf()

#Create values
acc_values = history.history['acc']
val_acc_values = history.history['val_acc']
epochs = range(1, len(acc_values) + 1)

#Create plot
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# COMMAND ----------

#Train the Model
################
#Network seems to overfit after 10 epochs (see graphs)
model.compile(optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size=512)

results_optim_epoch = model.evaluate(x_test, y_test)

# COMMAND ----------

results_optim_epoch

# COMMAND ----------


