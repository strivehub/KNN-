#coding:utf-8

from numpy import *
import operator


##给出训练数据以及对应的类别
def createDataSet():  #取数据和对应的标签
    group = array([[1.0,101.0], [5.0, 89.0], [115.0, 8.0], [108.0, 5.0]])  #训练样本
    labels = ['爱情片', '爱情片', '动作片', '动作片']   #数据对应标签
    return group, labels 


###通过KNN进行分类
def classify(input, dataSet, label, k):
    dataSize = dataSet.shape[0]
    ####计算欧式距离
    diff = tile(input, (dataSize, 1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff, axis=1)  ###行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5

    ##对距离进行排序
    sortedDistIndex = argsort(dist)  ##argsort()根据元素的值从大到小对元素进行排序，返回下标

    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  #计算每一类出现次数
    ###选取出现的类别次数最多的类别
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes

if __name__=="__main__":
    dataSet, labels = createDataSet()
    input = array([2.0, 120.0])  #待测试样本
    K = 1  #最近邻则K是取1
    output =classify(input, dataSet, labels, K)
    print("测试数据为:", input, "分类结果为：", output)
