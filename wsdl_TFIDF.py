from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import numpy as np
path = '../data/qos/wsdl'

wsdl_docs=[]
filenames = os.listdir(path)  #获取路径下的所有文件名

for i in range(len(filenames)):
    filename=filenames[i]
    file=open(path+'/'+filename,'r',encoding='ISO-8859-1')
    wsdl_doc=file.read()
    wsdl_docs.append(wsdl_doc)

wsdl_docs=[word_tokenize(wsdl) for wsdl in wsdl_docs] #对每个句子进行分词
wsdl_docs = [' '.join(wsdl) for wsdl in wsdl_docs]
print(type(wsdl_docs[0]))

print('分词完成')
# 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer(max_features=4)
# 该类会统计每个词语的tf-idf权值
tf_idf_transformer = TfidfTransformer()
# 将文本转为词频矩阵并计算tf-idf
tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(wsdl_docs))
# 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
x_train_weight = tf_idf.toarray()

np.savetxt('wsdl_TFIDF_dim32',x_train_weight)
print('输出x_train文本向量：')
print(x_train_weight)

