## 利用pyspark-sumbit进行提交代码，分布式计算
#/spark/spark-2.1.0-bin-hadoop2.7/bin/spark-submit --master spark://192.168.10.4:7077 --num-executors 2 --driver-memory 1g --executor-memory 2g --executor-cores 2 /spark/daima/xsaj.py 100

## pyspark 进行相似度计算，可分为数据部分和文字部分
##设置环境变量
##设置环境变量  

#encoding:utf-8
from pyspark import SparkConf,SparkContext
from pyspark.sql  import SQLContext
conf = SparkConf().setAppName('super')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

##加载所需PCA、Kmeans包
from pyspark.ml.feature import PCA,VectorAssembler
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import Vectors,VectorUDT 
from pyspark.ml.feature import * 
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.feature import StandardScaler
from pyspark.sql.functions import udf
from pyspark.mllib.linalg import DenseMatrix
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.feature import PCA
from pyspark.sql import SQLContext
import numpy as np
from numpy import array

# 读取数据
data = sc.textFile("/spark/bigdata/xsaj_jbxx_5000.txt")
#取出第一行的标题
header = data.first().split('\t')
#去掉第一行的标题
datas = data.map(lambda x:x.split('\t')).filter(lambda x : x != header)

## ---------------  数字部分计算pca进行降维  -----------------------------------------------------------------------------------------------------------------
##从数据中截取数字部分字段
test2 = datas.map(lambda x:x[6:])

### 利用自定义函数去掉数据中空格，并转成float型	
def rnul(str):
     is_na_list = []
     for i in range(len(str)):
         e = 0.0 if str[i]== '' else float(str[i].strip())
         is_na_list.append(e)
     return is_na_list
	 
test = test2.map(lambda x: rnul(x))

## rdd数据标准化
##对rdd操作
stdd1 = StandardScaler(True,True)
stdd = stdd1.fit(test)
resl = stdd.transform(test)

## pca mllib做主成分分析
model = PCA(10).fit(resl)
pcArray = model.transform(resl)
#pcArray.take(5)
data_pca = pcArray.map(lambda x:[x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]])
#data_pca.take(2)

## 合并编号及数据部分
ajbh = datas.map(lambda x:[x[2]])
data_part1 = ajbh.zip(data_pca)
data_part2 = data_part1.flatMap(lambda x:[x[0][0:]+x[1][0:]])

## 转成dataframe格式
from pyspark.sql import Row
from pyspark.sql.types import *
## 使用反射推断 Schema,定义rdd列名
data_part = data_part2.map(lambda p: Row(ajbh=p[0], d1=float(p[1]),d2=float(p[2]),d3=float(p[3]),d4=float(p[4]),d5=float(p[5]),d6=float(p[6]),d7=float(p[7]),d8=float(p[8]),d9=float(p[9]),d10=float(p[10])))
## 将rdd转成df
data_part_df = sqlContext.createDataFrame(data_part)
#data_part_df.select("ajbh","d1").show(2)

## 将文本部分数据进行缓存
data_part_df.cache()

## 将数据部分注册成表形式
data_part_df.registerTempTable("data_part")
#sqlContext.sql("select * from data_part").show(2)

## ---------------  文字部分计算转成词向量，并pca降维 ---------------------------------------------------------------------------------------------------------------

#使用ml
from pyspark.ml.feature import Word2Vec
import jieba
from pyspark.sql import SQLContext
from pyspark.sql import *
from pyspark.sql.types import *

#取出文本部分
#jyaq = datas.map(lambda x:x[5])

###名称，编号，文本部分
ds = datas.map(lambda x:[x[2],x[0],x[4],x[5]])

#加载用户词典
jieba.load_userdict('/spark/package/jieba-0.39/jieba/user_dict.txt')

#加载停用词词典
stopwords = [line.strip() for line in open('/spark/package/jieba-0.39/jieba/stopkey.txt','r').readlines()]

################Function###############
#过滤停用词
def swfilter(words ):
    fclist = []
    for word in words:
        if word not in stopwords:
            fclist.append(word)
    return fclist
#######################################
def coblist(list,str):
    l = []
    return l.append(list).append(str)

#分词
ds_fenci = ds.map(lambda w:[w[0],w[1],swfilter(jieba.lcut(w[2])),swfilter(jieba.lcut(w[3]))]).map(lambda x:[x[0],[x[1]]+x[2]+x[3]])
dsdf = sqlContext.createDataFrame(ds_fenci,['ajbh','fc'])

#定义模型
w2v = Word2Vec(vectorSize=30,seed=42,inputCol="fc",outputCol="model")
#取分词后的文本部分
fc_col = dsdf.select("fc")
#训练模型
model = w2v.fit(fc_col)
#将文本转换成向量
trs = model.transform(fc_col)

#转成rdd
trs_rdd = trs.rdd

#合并编号及文本部分
data_vec = ds_fenci.map(lambda x:x[0]).zip(trs_rdd.map(lambda x : x[1])).map(lambda x:[x[0], str(x[1][0]),str(x[1][1]),str(x[1][2]),str(x[1][3]),str(x[1][4]),str(x[1][5]),str(x[1][6]),str(x[1][7]),str(x[1][8]),str(x[1][9]),str(x[1][10]),str(x[1][11]),str(x[1][12]),str(x[1][13]),str(x[1][14]),str(x[1][15]),str(x[1][16]),str(x[1][17]),str(x[1][18]),str(x[1][19]),str(x[1][20]),str(x[1][21]),str(x[1][22]),str(x[1][23]),str(x[1][24]),str(x[1][25]),str(x[1][26]),str(x[1][27]),str(x[1][28]),str(x[1][29])])

#转成dataframe
data_vec_df = sqlContext.createDataFrame(data_vec,["ajbh","vec1","vec2","vec3","vec4","vec5","vec6","vec7","vec8","vec9","vec10","vec11","vec12","vec13","vec14","vec15","vec16","vec17","vec18","vec19","vec20","vec21","vec22","vec23","vec24","vec25","vec26","vec27","vec28","vec29","vec30"])
## 将文本部分数据进行缓存
data_vec_df.cache()

## 将文本部分注册成新表
data_vec_df.registerTempTable("data_vec")
#sqlContext.sql("select * from data_vec").show(2)

## ---------------  将所有数据进行合并，并pca降维 ------------------------------------------------------------------------------------------------------

## 根据编号 将数据部分和文本部分合并
data_all_df = sqlContext.sql("select distinct b.ajbh,a.d1,a.d2,a.d3,a.d4,a.d5,a.d6,a.d7,a.d8,a.d9,a.d10,b.vec1,b.vec2,b.vec3,b.vec4,b.vec5,b.vec6,b.vec7,b.vec8,b.vec9,b.vec10,b.vec11,b.vec12,b.vec13,b.vec14,b.vec15,b.vec16,b.vec17,b.vec18,b.vec19,b.vec20,b.vec21,b.vec22,b.vec23,b.vec24,b.vec25,b.vec26,b.vec27,b.vec28,b.vec29,b.vec30 from data_part a inner join data_vec b on a.ajbh=b.ajbh")

## data_all_df.show(2)
## 将dataframe转成rdd
data_all_rdd1 = data_all_df.rdd

## 将list转成vector (重要一步，df做算法是基于vector的，所以一定要将list转换成vector)
## list转换vector 的方法 ： data.map(lambda x:Vectors.dense(x),VectorUDT())
data_all = data_all_rdd1.map(lambda x:[x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20],x[21],x[22],x[23],x[24],x[25],x[26],x[27],x[28],x[29],x[30],x[31],x[32],x[33],x[34],x[35],x[36],x[37],x[38],x[39],x[40]]).map(lambda x:Vectors.dense(x),VectorUDT())
# data_all.take(2)
## 合并编号及数据部分、文本部分
data_all_final = data_all_rdd1.map(lambda x:x[0]).zip(data_all)
## data_all_final.take(2)
## 编程方式定义列名
data_all_ml = data_all_final.map(lambda p: Row(ajbh=p[0],features=p[1]))
## data_all_ml.take(2)
data_all_ml_df = sqlContext.createDataFrame(data_all_ml)
## 将所有的df数据进行缓存
data_all_ml.cache()
## 将数据结果转成dataframe，可以暂时不用
data_all_ml_df.registerTempTable("data_all")
#sqlContext.sql("select * from data_all").show()

## 做pca降维
from pyspark.ml.feature import PCA
## 定义pca维度
pca = PCA(k=30, inputCol="features", outputCol="pca_features")
## 训练pca模型
model = pca.fit(data_all_ml_df)
## 将pca结果转换成df
data_all_pca = model.transform(data_all_ml_df)
## 将pca结果缓存
data_all_pca.cache()
#data_all_pca.show(2,truncate = False)

## ---------------  将所有数据pca降维结果进行聚类分析  ------------------------------------------------------------------------------------------------------

## 对pca结果进行聚类分析
from pyspark.ml.clustering import KMeans
## 定义kmeans形式
kmeans = KMeans(featuresCol="pca_features", predictionCol="prediction", k=300, initMode="k-means||", initSteps=2, tol=1e-4, maxIter=20, seed=1234)
## 对样本数据进行训练
model = kmeans.fit(data_all_pca)
## 找出聚类中心
#centers = model.clusterCenters()
#len(centers)
## 将聚类结果转成df
data_all_cluster = model.transform(data_all_pca)
# data_all_cluster.show(2)
## 将聚类结果进行缓存
data_all_cluster.cache()

## 将聚类结果注册成表
data_all_cluster.registerTempTable("data_all_cluster")
#sqlContext.sql("select * from data_all_cluster").show(2)

## ---------------  将所有数据聚类结果计算余弦相似度  ------------------------------------------------------------------------------------------------------

## 筛选出余弦相似度的所需字段
data_cos = sqlContext.sql("select (row_number() over(order by ajbh)) id,ajbh,prediction cluster_id,pca_features from data_all_cluster")
## 将余弦相似度数据转成rdd
data_cos_rdd = data_cos.rdd
## 将余弦相似度数据拆分成多列
data_cos_all = data_cos_rdd.map(lambda x:[x[0],x[1],x[2],x[3][0],x[3][1],x[3][2],x[3][3],x[3][4],x[3][5],x[3][6],x[3][7],x[3][8],x[3][9],x[3][10],x[3][11],x[3][12],x[3][13],x[3][14],x[3][15],x[3][16],x[3][17],x[3][18],x[3][19],x[3][20],x[3][21],x[3][22],x[3][23],x[3][24],x[3][25],x[3][26],x[3][27],x[3][28],x[3][29]])

## 将整理出的多列rdd转成df
# 用row形式定义数据字段schema，记得这里要转换，定义数据类型为 df支持的类型
data_cos_row = data_cos_all.map(lambda p: Row(id=p[0],ajbh=p[1],cluster_id=p[2],d1=float(p[3]),d2=float(p[4]),d3=float(p[5]),d4=float(p[6]),d5=float(p[7]),d6=float(p[8]),d7=float(p[9]),d8=float(p[10]),d9=float(p[11]),d10=float(p[12]),d11=float(p[13]),d12=float(p[14]),d13=float(p[15]),d14=float(p[16]),d15=float(p[17]),d16=float(p[18]),d17=float(p[19]),d18=float(p[20]),d19=float(p[21]),d20=float(p[22]),d21=float(p[23]),d22=float(p[24]),d23=float(p[25]),d24=float(p[26]),d25=float(p[27]),d26=float(p[28]),d27=float(p[29]),d28=float(p[30]),d29=float(p[31]),d30=float(p[32])))
# 转成df
data_cos_df = sqlContext.createDataFrame(data_cos_row)
# data_cos_df.show(2)

## 将聚类结果注册成表
data_cos_df.registerTempTable("data_cos")
#sqlContext.sql("select * from data_cos").show(2)

## 利用sparksql计算相似度
cosine = spark.sql("select * from(select a.ajbh ajbh_a,b.ajbh ajbh_b,a.cluster_id cluster_id_a,b.cluster_id cluster_id_b,a.id id_a,b.id id_b,((a.d1*b.d1+a.d2*b.d2+a.d3*b.d3+a.d4*b.d4+a.d5*b.d5+a.d6*b.d6+a.d7*b.d7+a.d8*b.d8+a.d9*b.d9+a.d10*b.d10+a.d11*b.d11+a.d12*b.d12+a.d13*b.d13+a.d14*b.d14+a.d15*b.d15+a.d16*b.d16+a.d17*b.d17+a.d18*b.d18+a.d19*b.d19+a.d20*b.d20+a.d21*b.d21+a.d22*b.d22+a.d23*b.d23+a.d24*b.d24+a.d25*b.d25+a.d26*b.d26+a.d27*b.d27+a.d28*b.d28+a.d29*b.d29+a.d30*b.d30)/((sqrt(a.d1*a.d1+a.d2*a.d2+a.d3*a.d3+a.d4*a.d4+a.d5*a.d5+a.d6*a.d6+a.d7*a.d7+a.d8*a.d8+a.d9*a.d9+a.d10*a.d10+a.d11*a.d11+a.d12*a.d12+a.d13*a.d13+a.d14*a.d14+a.d15*a.d15+a.d16*a.d16+a.d17*a.d17+a.d18*a.d18+a.d19*a.d19+a.d20*a.d20+a.d21*a.d21+a.d22*a.d22+a.d23*a.d23+a.d24*a.d24+a.d25*a.d25+a.d26*a.d26+a.d27*a.d27+a.d28*a.d28+a.d29*a.d29+a.d30*a.d30)*sqrt(b.d1*b.d1+b.d2*b.d2+b.d3*b.d3+b.d4*b.d4+b.d5*b.d5+b.d6*b.d6+b.d7*b.d7+b.d8*b.d8+b.d9*b.d9+b.d10*b.d10+b.d11*b.d11+b.d12*b.d12+b.d13*b.d13+b.d14*b.d14+b.d15*b.d15+b.d16*b.d16+b.d17*b.d17+b.d18*b.d18+b.d19*b.d19+b.d20*b.d20+b.d21*b.d21+b.d22*b.d22+b.d23*b.d23+b.d24*b.d24+b.d25*b.d25+b.d26*b.d26+b.d27*b.d27+b.d28*b.d28+b.d29*b.d29+b.d30*b.d30)))) sim from data_cos a cross join data_cos b where a.ajbh <> b.ajbh and a.cluster_id=b.cluster_id and a.id>b.id) aa where aa.sim > 0.5")
## 查看cosine值
# cosine.show(5)
cosine.cache()
#保存成json
cosine.write.format('json').save("/spark/bigdata/out_all_5000")    ## 分多个文件存储，不合并
## cosine.coalesce(1).write.format('json').save("hdfs:///usr/isw_6/ml/out/jyaq_300w")   ##将文件合并成一个再输出
