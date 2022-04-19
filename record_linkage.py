import pyspark.sql.functions  as f
import pandas as pd 
import pyspark.sql.types
from pyspark.sql import Row
import numpy as np


sc.setLogLevel('OFF')
SparkContext.setSystemProperty('spark.executor.memory','4g')
SparkContext.setSystemProperty('spark.driver.memory','4g')
SparkContext.setSystemProperty('spark.master','local[*]')
SparkContext.setSystemProperty("spark.scheduler.mode", "FAIR")
"""  Read the data into a pyspark dataframe. We may opt to the DataFrameReader API inferring the schema for the data set, however this requires reading the data twice.
  Alternatively, If the exact schema of the dataset is known, it may be supplied through the 'schema' method of the Spark DataFrameReader API, hence saving resources when 
  reading the data sets."""
print("***   Read the data")
#Clean your workspace before executing the code just to be sure that there are no synonymous variables already in your work space. 
pr_df = spark.read.option('header','true').option('inferSchema','true').option('nullValue','?').format('csv').load('hdfs://localhost:9000/user/data/blocks')
pr_df.cache()

# Data exploration

print("***   Data Exploration")
print(f"Number of rows: {pr_df.count()}")
print(f"Data set schema: \n ")
pr_df.printSchema()
print("The first 5 rows:")
pr_df.show(5)
   

def round_df(df):
    new_df = spark.createDataFrame(df.rdd)
    new_df.cache()
    cols = new_df.columns
    for col in cols:
        if (str(new_df.schema[col].dataType) == 'FloatType') | (str(new_df.schema[col].dataType) == 'DoubleType'):
            new_df = new_df.withColumn(col, f.round(col,3))
    return new_df 
    
#Round to the third decimal digit the fields of collumns that accept float or double values.
pr_df_view = round_df(pr_df)
pr_df_view.cache()  
#pr_df_view.show(10)
schema = pr_df.schema

print("Summary description of the data set:") 
summary = pr_df.describe()

def convert_types(df, schema = None, cols = None, types = None):
    if schema != None:
        for f in schema:
            if f.name in df.columns:
                df = df.withColumn(f.name, df[f.name].cast(f.dataType)) 
    elif (cols != None) & (types != None):
        for i,col in enumerate(cols):
            df = df.withColumn(col, df[col].cast(types[i]))
    return df

#Change the schema of the summary df
# These statictics are computed excluding the NaN values.
#summary = convert_types(summary, cols, ['float']* len(cols))
summary = convert_types(summary, schema = schema )
summary.printSchema()

round_df(summary).show()
# From the summary view of the data we may extract several observations.
print("""\n  From the summary data, we observe that less than 2% of the records have non-null values in the 
  'cmp_fname_c2' field. For using a column as a feature for our classifier, we expect that this column 
  contains values for most instances, except for the case, where the absence of a value for an attribute
  bears some meaning, for the purpose of the classifier.""")

print("\nHow is each class represented in the dataset? Is the dataset balanced?")
pr_df.groupBy('is_match').count().orderBy(f.desc('count')).show()
# or pr_df.rdd.map(lambda r : r['is_match']).countByValue()
# In the case that we are using the DataFrame API for the aggregation, is more efficient and faster as the Spark 
# engine takes an optimized decision on how to perform the aggregation.

print("\n  More analytics...")
pr_df.agg(f.sum('cmp_fname_c1'), f.avg('cmp_lname_c1'), f.mean('cmp_fname_c1'),f.mean('cmp_lname_c1'), f.stddev('cmp_fname_c1'), f.stddev('cmp_lname_c1'),f.stddev_pop('cmp_fname_c1'), f.stddev_pop('cmp_lname_c1')).show()

print("How is each attribute correlated to each of the two classes? ")
matches_st = pr_df.where("is_match == 'true'").describe()
mis_matches_st = pr_df.where(pr_df.is_match == f.lit('false')).describe()


print("Lets carry out some statistics that will help in identifying the most determinant attributes that can be selected as features for the prediction model.")
print("""  For this purpose the 'matched_st' and the 'mis_matched_st' dataframes are pivoted (having as column attributes, the 'count', the 'mean', the 'stddev' etc) and then
  joined together on the 'feature_name', performing specific comparisons between the statistics of the columns. """)

def pivot_df(ms):
    cols = ms.columns
    long_ms = ms.rdd.flatMap(lambda r: [[r['summary'], cols[i], float(k)] for i,k in enumerate(r) if i>0 ] )
    ms_piv = long_ms.toDF(["metric","feature_name",'value'])
    pivoted_ms = ms_piv.groupBy('feature_name').pivot("metric",["count","mean","stddev","min","max"]).agg(f.first('value'))
   
    return pivoted_ms
 

pms = pivot_df(matches_st)
pms.show()

pmms = pivot_df(mis_matches_st)
pmms.show()

pms.createOrReplaceTempView('piv_matches_sum')
pmms.createOrReplaceTempView('piv_mmatches_sum') 
res = spark.sql("select a.feature_name, a.count + b.count total, a.mean - b.mean diff_of_means, (a.count - b.count)/(a.count + b.count) balance_idx from piv_matches_sum a inner join piv_mmatches_sum b on a.feature_name == b.feature_name where a.feature_name not in ('id_1','id_2') order by total desc, diff_of_means desc, abs(balance_idx) asc") 
res.show()
print("""  total = a.count + b.count \n  diff_of_means = a.mean - b.mean\n  balance_idx = (a.count - b.count)/(a.count + b.count)   """)
print("""  A good feature should exhibit at least two properties: first a small percentage of null values, so that it regularly occurs and hence it is reliable
  for being considered along with the other features in our analysis and secondly, the delta value (difference between the mean values) is as large as possible 
  in the range [0,1] so that the specific feature has the ability to dicriminate data between the two classes.
  From the table above, 'cmp_plz','cmp_by' and 'cmp_bd' match best the above characteristics and hence are prominent features to select for the prediction model.
  Features 'cmp_lname_c1' and 'cmp_bm' seem beneficial too, having high rate of appearance and showing substatial difference between the mean values of the matched 
  and unmatched data.""")

print("  A simplistic model is created, which attributes as score to each new record, the sum of the values corresponding to the features mentioned above.")  

def score_value(v):
    if v is None:
        return int(0)
    else:
        return v
    
sc_df = pr_df.rdd.map(lambda r : Row(score = score_value(r['cmp_lname_c1']) + score_value(r['cmp_by']) + score_value(r['cmp_bd']) + score_value(r['cmp_bm']) + score_value(r['cmp_plz']),is_match = r['is_match'])).toDF(['score','is_match'])  

sc_df.show()

sc_df.cache()

print("  Which is the appropriate threshold to select for eliminating the possibility of getting many false positives and false negatives?")
max_score = sc_df.agg(f.max('score').alias('max_score')).collect()[0].max_score
sc_df_nor = sc_df.rdd.map(lambda r : Row(score_nor = r.score/max_score, score = r.score, is_match = r.is_match, is_match_quan = 1 if r.is_match == True else 0)).toDF(['score_nor', 'score', 'is_match','is_match_quan'])
sc_df_nor.cache()
print("  We choose to normalize the 'score', by dividing it with the maximum score value, and to quantify the 'is_match' column with 1 if its value is True and 0 if it is False.")
sc_df_nor.show()
print("  We choose to calculate the difference between the normalized score and the quantified 'is_match' column and take quantiles. From the quantiles, we can have a \
  view on how well the model is doing and obtain a possible score value to use as a threshold, which will yield to a decreased number of false estimates. Then, by trying\
  more values in the region of the threshold picked, we can refine our estimates and balance accordingly the false positives and false negatives.")
score_dif =sc_df_nor.rdd.map(lambda r : Row(score_dif = float(np.absolute(r.is_match_quan - r.score_nor )))).toDF(['score_dif'])
score_dif.cache()
q = score_dif.stat.approxQuantile('score_dif',[0.25, 0.5,0.90],0.05)
print(f"  Approximate quantiles: {q}")
print("  A 90% of the population was assigned a score that diverges from the real by approximately 0.4, which means that in the case of a match it \
  favors the match scenario and in the case of a mismatch, it favors the mismatch scenario. So (1 - 0.4)*maxVar = 3 is an indicative threshold to \
  begin with.")

print("  Congestion matrix for threshold = 3")
cross_tab = sc_df.selectExpr("score >3 as above","is_match").groupBy('above').pivot('is_match',['true','false']).count()
cross_tab.cache()
cross_tab.show()
print(f"fp + fn = {cross_tab.select('false').where('above = true').collect()[0].false+cross_tab.select('true').where('above = false').collect()[0].true}")     
print("  The false positive examples are more than the false negative ones.")
print("  Congestion matrix for threshold = 4.8")
cross_tab1 = sc_df.selectExpr("score >4.8 as above","is_match").groupBy('above').pivot('is_match',['true','false']).count()
cross_tab1.cache()
cross_tab1.show()
print(f"fp + fn = {cross_tab1.select('false').where('above = true').collect()[0].false+cross_tab1.select('true').where('above = false').collect()[0].true}")  
print("  The false positive examples are fewer than the false negative ones.")
print("  Congestion matrix for threshold = 3.888")
cross_tab2 = sc_df.selectExpr("score >3.888 as above","is_match").groupBy('above').pivot('is_match',['true','false']).count()
cross_tab2.cache()
cross_tab2.show()
print(f"fp + fn = {cross_tab2.select('false').where('above = true').collect()[0].false+cross_tab2.select('true').where('above = false').collect()[0].true}")  
print("  The sum of the false positive plus the false negative instances, in this case, is the smallest that we have noticed.")


from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import BinaryClassificationEvaluator



cols = ['cmp_plz','cmp_by','cmp_bd','cmp_lname_c1' ,'cmp_bm','is_match']

prn = pr_df.select(cols).dropna()
sub = prn.rdd.map(lambda r : Row(Vectors.dense(r['cmp_plz']),Vectors.dense(r['cmp_by']),Vectors.dense(r['cmp_bd']),Vectors.dense(r['cmp_lname_c1']) ,Vectors.dense(r['cmp_bm']), 1 if  r['is_match'] == True else 0)).toDF(cols)

scaler = []
for col in cols[:-1] :
    if scaler == None:
        scaler = [StandardScaler(inputCol = col, outputCol = col+"_sc", withMean=True, withStd=True)]
    else:
        scaler.append(StandardScaler(inputCol = col, outputCol = col+"_sc", withMean=True, withStd=True))
    
va = VectorAssembler(inputCols = [col+"_sc" for col in cols[:-1]], outputCol = 'features')
pipe = Pipeline()
pipe.setStages(scaler+[va])
pipe_model = pipe.fit(sub)
ds = pipe_model.transform(sub)
ds = ds.select(['features','is_match'])

lr = LogisticRegression(labelCol = 'is_match')
train, test = ds.randomSplit([0.8, 0.2], seed = 27)
lrm = lr.fit(train)
predictions = lrm.transform(test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol = 'prediction', labelCol = 'is_match')
score = evaluator.evaluate(predictions)
print(f"  The auc score on the test set is : {score}")
print("\n  The confusion matrix is :  ")
conf_matrix  = predictions.withColumn('conf_col',f.udf(lambda pred, label : 'tp' if pred == label == 1 else 'tn' if pred == label == 0 else 'fn' if (pred == 0) & (pred != label) else 'fp'  )(f.col('prediction'), f.col('is_match'))).groupBy('conf_col').count()
conf_matrix.show()