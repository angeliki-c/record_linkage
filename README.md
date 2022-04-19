# record_linkage


Record Linkage

	A problem we are often called upon to address, when dealing with big volumes of data, often at 
	the data cleansing phase of the data processing pipeline, is that of identifying the records
	that essentially correspond to the same entities. Upon their recognition, we act according to
	the policy adopted in each case we study (e.g delete the records, merge records e.t.c.).
  	We encounter the need of entity resolution, specially when there is information coming from 
	multiple sources about the entities, represented by a different set of attributes depended on 
	the source from which the information originated.
 
 	In the literature this challenge is often encountered with different names, such as:
	Record deduplication, Entity resolution, Merge and purge, List washing, Record linkage and 
	others
 
 
 
Techniques followed

	In our case study, a simplistic model is created, in a much similar way as the one folowed in [1],
	for the prediction of whether two entities contained in each record of a database match (it is the
	same entity). Subsequently, a simple evaluation approach is used, as one would be applied, when being
	at the early stage of our data problem analysis, as our aim in this work has not been, on developing
	the most effective entity resolution method. However, from a comparison with the performance of a 
	logistic regression model for the entity matching, following the observations from this early analysis,
	which is quite good, it seems that this analytical approach brings us closer to the way we should start
	building the model.
	
 


Data set

	The data set used in this case study [2] corresponds to a sample data set, from the UC Irvine Machine 
	Learning Repository, which has been created for research on the record linkage task. It consists of 
	several million pairs of patient records that were matched according to several different criteria, 
	such as the patient’s name (first and last), address, and birthday. Each field has been assigned a 
	numerical score from 0.0 to 1.0, based on how similar the strings between the two entities in that field
	were and the data was then hand-labeled to identify, which pairs represented the same persons and which 
	did not. 
	
	Number of rows: 5749132
	This is relatively a small dataset and it is expected to fit and be analyzed locally or on one node of 
	your cluster.



Challenges

	The value of an attribute across records that correspond to the same entity might not always be correct 
	and the same for all records. Conversaly, the values in some attributes across records that correspond 
	to different entities might be the same, despite the fact that correspond to different entities.
	The value of an attribute across records that correspond to the same entity might have different formatting
	or typos, which makes it difficult to apply a simple equality check approach for the entity resolution
	problem. In general, there should be multiple criteria for matching, playing at the same time in a volume 
	of records.



Evaluation

	The classification capability of the model is tested against the number of false negatives and the number
	of false positives.


 
Code

  record_linkage.py
   
  The code has been tested on a Spark standalone cluster. For the Spark setting, spark-3.1.3-bin-hadoop2.7  
  bundle has been used.   
  All can be run interactively with pyspark shell or by submitting  
      e.g. exec(open("project/location/record_linkage/record_linkage.py").read()) for an all at once execution.  
  The external python packages that are used in this implementation exist in the requirements.txt file.   
  Install with:   
	    pip install -r project/location/record_linkage/requirements.txt
  This use case is inspired from the series of experiments presented in [1], though it deviates from it, in the  
  programming language, the setting used and in the analysis followed.
  
 


References

    1. Advanced Analitics with Spark-Patterns for Learning from Data at Scale【Sandy Ryza, Uri Laserson, Sean Owen, Josh Wills】(2017)
    2. https://bit.ly/1Aoywaq
	
	
