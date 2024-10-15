#Indigo Catton
# CS 777 Final Project

import sqlite3
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, VectorSlicer
from pyspark.ml.feature import OneHotEncoder, UnivariateFeatureSelector
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


# Connect to sqlite database
fia = sqlite3.connect("C:\\Users\\indig\\Documents\\CS777\\FinalProject\\SQLite_FIADB_UT\\SQLite_FIADB_UT.db")
cursor_obj = fia.cursor()
  
# Query 
query = '''SELECT PLOT.CN,
                ELEV,
                SEEDLING.TREECOUNT,
                COUNT(TREE.CN) AS TREE_CT,
                SUM(TREE.HT) AS HT_SUM,
                SUM(TREE.DIA) AS DIA_SUM,
                SUM(TREE.CARBON_AG) AS CARBON_AG_SUM,
                SUM(TREE.CARBON_BG) AS CARBON_BG_SUM,
                SUM(TREE.CR) AS CR_SUM,
                SUM(TREE.WDLDSTEM) as WDLD_SUM, 
                COND.SLOPE,
                COND.ASPECT,
                COND.STDAGE,
                COND.BALIVE,
                COND.ALSTK,
                COND.CARBON_DOWN_DEAD,
                COND.CARBON_LITTER,
                COND.CARBON_SOIL_ORG,
                COND.CARBON_UNDERSTORY_AG,
                COND.CARBON_UNDERSTORY_BG,
                COND.CONDID,
                COND.COND_STATUS_CD,
                COND.FORTYPCD,
                COND.SITECLCD
                FROM PLOT
                JOIN SEEDLING ON PLOT.CN = SEEDLING.PLT_CN
                JOIN TREE ON PLOT.CN = TREE.PLT_CN
                JOIN COND on PLOT.CN = COND.PLT_CN
                GROUP BY PLOT.CN
                '''

# Run quer
cursor_obj.execute(query) 
output = cursor_obj.fetchall() 

# Close the connection
fia.close()

# Count None values for each column
transposed_data = list(zip(*output))
null_counts = [sum(1 for x in col if x is None) for col in transposed_data]

# Print null counts
for col_index, null_count in enumerate(null_counts):
    print(f"Col {col_index}: {null_count} nulls")

# Initialize Spark session
spark = SparkSession.builder.appName("Project").getOrCreate()

# Create an df from the fetched results
df = spark.sparkContext.parallelize(output).toDF(["PLOT_CN",
                "ELEV",
                "SEEDLING_TREECOUNT",
                "TREE_CT",
                "HT_SUM",
                "DIA_SUM",
                "CARBON_AG_SUM",
                "CARBON_BG_SUM",
                "CR_SUM",
                "WDLD_SUM",
                "SLOPE",
                "ASPECT",
                "STDAGE",
                "BALIVE",
                "ALSTK",
                "CARBON_DOWN_DEAD",
                "CARBON_LITTER",
                "CARBON_SOIL_ORG",
                "CARBON_UNDERSTORY_AG",
                "CARBON_UNDERSTORY_BG",
                "CONDID", #category
                "COND_STATUS_CD", #category
                "FORTYPCD", #category
                "SITECLCD" #category
                ])
# Fill null values with 0 for seedling count column
df_filled = df.fillna(0, subset=["SEEDLING_TREECOUNT"])

# Filter out rows with null vlues
df_filtered = df_filled.na.drop()
print(df_filled.count())
print(df_filtered.count())

#Combine biomass values to create target
df_with_sum = df_filtered.withColumn("BIOMASS", df_filtered.CARBON_AG_SUM + df_filtered.CARBON_BG_SUM)

# One hot encoding for categorical variables
encoder = OneHotEncoder(inputCols=["CONDID", "COND_STATUS_CD", "FORTYPCD", "SITECLCD" ],
                        outputCols=["CONDID_VEC", "COND_STATUS_CD_VEC", "FORTYPCD_VEC", "SITECLCD_VEC"])

model = encoder.fit(df_with_sum)
encoded = model.transform(df_with_sum)

# Combine features to one column
inputCols = ["ELEV","SEEDLING_TREECOUNT", "TREE_CT","HT_SUM","DIA_SUM","CR_SUM","WDLD_SUM","SLOPE",
            "ASPECT","STDAGE","BALIVE","ALSTK","CARBON_DOWN_DEAD","CARBON_LITTER", "CARBON_SOIL_ORG",
            "CARBON_UNDERSTORY_AG","CARBON_UNDERSTORY_BG","CONDID_VEC","COND_STATUS_CD_VEC","FORTYPCD_VEC",
            "SITECLCD_VEC"]

assembler = VectorAssembler(inputCols= inputCols, outputCol="all_features")
vectorized = assembler.transform(encoded)

# Split into training and testing
train, test = vectorized.randomSplit([0.8, 0.2], seed=10)

# Scale with data with standard scaling
scaler = StandardScaler(inputCol="all_features", outputCol="features", withMean=True, withStd=True)
scalerModel = scaler.fit(train) # fit on testing data
train_scaled = scalerModel.transform(train)
test_scaled = scalerModel.transform(test) 

# Initialize a linear regression model
lr = LinearRegression(featuresCol='features', labelCol='BIOMASS', regParam=0.3, maxIter=10) 

# Train the model on training data
model = lr.fit(train_scaled)

# Make predictions on the test set
predictions = model.transform(test_scaled)

# evaluate the model
rmse_evaluator = RegressionEvaluator(labelCol='BIOMASS', predictionCol='prediction', metricName='rmse')
r2_evaluator = RegressionEvaluator(labelCol='BIOMASS', predictionCol='prediction', metricName='r2')
rmse = rmse_evaluator.evaluate(predictions)
r2 = r2_evaluator.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE) = {rmse}, R2 = {r2}")

# Print intercept
print("Intercept: %s" % str(model.intercept))

# With feature selection 
# lasso regualrization for feature importance
lasso = LinearRegression(elasticNetParam=1.0, regParam=1.0, maxIter=100,featuresCol="features", labelCol="BIOMASS")
lasso_model = lasso.fit(train_scaled)

# get lasso regression coefficients
coefficients = lasso_model.coefficients

# Get indicies with non 0 coefficients
indices = [idx for idx, val in enumerate(coefficients) if val != 0]
top_indices = sorted(range(len(coefficients)), key=lambda i: abs(coefficients[i]), reverse=True)[:10]
    
# Get selected features
slice = VectorSlicer(inputCol="features", outputCol="selected_features", indices = indices)
train_subset = slice.transform(train_scaled)
test_subset = slice.transform(test_scaled)

#Re-Run Linear Regression
# Train the model on training data
lr2 = LinearRegression(featuresCol='features',regParam=0.3, labelCol='BIOMASS', maxIter=10) 
model2 = lr2.fit(train_subset)

# Make predictions on the test set
predictions = model2.transform(test_subset)

# evaluate the model
rmse_evaluator = RegressionEvaluator(labelCol='BIOMASS', predictionCol='prediction', metricName='rmse')
r2_evaluator = RegressionEvaluator(labelCol='BIOMASS', predictionCol='prediction', metricName='r2')
rmse = rmse_evaluator.evaluate(predictions)
r2 = r2_evaluator.evaluate(predictions)

print(f"Feature Selection: Root Mean Squared Error (RMSE) = {rmse}, R2 = {r2}")

# Univariant Feature Selector top 5
selector = UnivariateFeatureSelector(featuresCol="features", outputCol="5Features",
                                     labelCol="BIOMASS", selectionMode="numTopFeatures")

selector.setFeatureType("continuous").setLabelType("continuous").setSelectionThreshold(5)

train_selected = selector.fit(train_scaled).transform(train_scaled)
test_selected = selector.fit(train_scaled).transform(test_scaled)
print(train_selected.head())
#selected_feature_indices = selector.selectedFeatures

#print("Top 5 features and coefficient")
#for i in selected_feature_indices:
#    print(f"Feature: {inputCols[i]}")
          
#Re-Run Linear Regression
# Train the model on training data
lr3 = LinearRegression(featuresCol='5Features',regParam=0.3, labelCol='BIOMASS', maxIter=10) 
model3 = lr3.fit(train_selected)

# Make predictions on the test set
predictions = model3.transform(test_selected)

# evaluate the model
rmse_evaluator = RegressionEvaluator(labelCol='BIOMASS', predictionCol='prediction', metricName='rmse')
r2_evaluator = RegressionEvaluator(labelCol='BIOMASS', predictionCol='prediction', metricName='r2')
rmse = rmse_evaluator.evaluate(predictions)
r2 = r2_evaluator.evaluate(predictions)

print(f"5 Feature Selection: Root Mean Squared Error (RMSE) = {rmse}, R2 = {r2}")


spark.stop()