from pyspark import SparkContext, SparkConf
import os
import sys
 
conf = SparkConf()
sc = SparkContext(appName="HW3", conf=conf)

graph = sc.textFile(str(sys.argv[3]))
twi_graph = graph.map(lambda x: x.split('\t')[::-1]).cache()

start = str(sys.argv[1])
end = str(sys.argv[2])

buf = twi_graph.filter(lambda x: x[0] == start).map(lambda x: (x[1], start))
buf = twi_graph.join(buf).map(lambda x: (x[1][0], (x[0], x[1][1])))
k = 0
while k <= 20:
    buf = twi_graph.join(buf).map(lambda x: (x[1][0], (x[0], ) + x[1][1]))
    if buf.keys().filter(lambda x: x == end).count() > 0:
        buf.filter(lambda x: x[0] == end).map(lambda x: ((x[0],) + x[1])[::-1])\
                                         .map(lambda x: ','.join(x))
        buf.saveAsTextFile(str(sys.argv[4]))
        break
    k += 1
