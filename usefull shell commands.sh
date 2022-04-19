# assign the desired permission to the folder of the project for accessing code + data
sudo chmod 700 -R the/location/of/your/project/record_linkage/
# start the ssh client and server
sudo service ssh --full-restart
# start hadoop
start-dfs.sh
# copy the data to hadoop file system
hdfs dfs -put record_linkage/data hdfs://localhost:9000/user/
# start pyspark 
pyspark