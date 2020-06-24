# -*- coding: utf-8 -*-
"""
Stub, copy and use to turn large CSV files into a Spark-Object
"""

from pyspark.sql import SparkSession
from pyspark.sql import Row

import collections

# Create a SparkSession.
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

def mapper(line):
    fields = line.split(',')
    Date=str(fields[0]),
    Open=str(fields[1]), 
    High=str(fields[2]), 
    Low=str(fields[3]), 
    Close=str(fields[4]),
    AdjClose=str(fields[5]),
    Volume=str(fields[6])

lines = spark.sparkContext.textFile("AAPL3.csv")
Stock = lines.map(mapper)

# Infer the schema, and register the DataFrame as a table.
schemaStock = spark.createDataFrame(Stock).cache()
schemaStock.createOrReplaceTempView("Stock")


schemaStock.head()


