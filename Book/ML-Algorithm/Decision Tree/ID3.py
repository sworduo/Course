import numpy as np
from plotDT import *

#这里有个小错误，假设有两个变量x1取值为1,2,3,x2取值为a，b，c，
#如果数据集中没有（1,a）的实例项，而预测的时候出现了这个实例项，那么预测时将会报错。
#因为本身决策树就没考虑这种情况。
#解决方法参考CART的实现。
#对于每一个x1取值v1下，x2所有未出现的取值v2的情况，一律将此时集合的投票作为（v1,v2）的判别结果。


#ID3决策树，不包含剪枝。
#1、判断是否只剩下一个类别。
#2、判断是否还有特征。
#3、寻找信息增益最大的特征。
#4、根据这个特征将原数据集划分为子集。
#5、重复1-4的过程。

#计算类别的熵。
#输入：dataset(数据集 list [[]])    其中每个实例最后一个元素代表相应的类别。
#输出：entropy(熵) 这个数据集类别的熵。
def calShannon(dataset):
    labeldict = {}
    datasize = len(dataset)
    for item in dataset :
        currentlabel = item[-1]
        if item[-1] not in labeldict :
            labeldict[currentlabel] = 0
        labeldict[currentlabel] += 1
    entropy = 0.0
    for key in labeldict :
        pro = float(labeldict[key])/datasize
        entropy -= pro * np.log2(pro)
    return entropy

#根据特征的不同取值划分子集。
#输入：  dataset   (数据集 list [[]])
#       featpos  (指明特征所在的列）
#       value   (指明根据该特征的哪个取值划分子集）
#输出：  subset (根据第i个特征的第j个取值所划分的子集。)
def splitDataset(dataset, featpos, value) :
    subset = []
    for item in dataset :
        if item[featpos] == value :
            line = item[:featpos]
            line.extend(item[featpos+1:])
            subset.append(line)
    return subset

#选择信息增益最大的特征。
#输入：    dataset 数据集 list [[]]
#输出：    featurepos  该特征所在的列。
def chooseBestFeatureToSplit(dataset):
    bestFeaturePos = -1
    bestInfoGain = 0
    dataEntropy = calShannon(dataset)
    featuresize = len(dataset[0])-1
    datasetsize = len(dataset)
    for featurepos in range(featuresize) :
        #确定这个特征有多少个取值。
        featureValueVec = [example[featurepos] for example in dataset]
        featureValueSet = set(featureValueVec)
        thisEntropy = 0.0
        for featureValue in featureValueSet :
            subset = splitDataset(dataset, featurepos, featureValue)
            pro = float(len(subset))/datasetsize
            subEntropy = calShannon(subset)
            thisEntropy += pro * subEntropy
        thisInfoGain = dataEntropy - thisEntropy
        if thisInfoGain > bestInfoGain :
            bestFeaturePos = featurepos
            bestInfoGain = thisInfoGain
    return bestFeaturePos

#投票占优类别。
#输入: dataset
#输出: bestlabel  数量最多的类别。
def vote(dataset) :
    labeldict = {}
    for label in dataset :
        if label not in labeldict.keys() :
            labeldict[label] = 0
        labeldict[label] += 1
    sortedLabel = sorted(labeldict.items(), key=lambda item:item[1], reverse=True)
    return sortedLabel[0][0]

#构建决策树。
#输入：dataset 数据集.
#   featurelabel [] 每个特征的名字，比如说年龄、有房与否等等.
#输出：Mytree 构建好的决策树.
def createDecisionTree(dataset, featurelabel) :
    labelvec = [example[-1] for example in dataset]
    #如果所有实例的类别都一样则返回。
    if labelvec.count(labelvec[0]) == len(labelvec) :
        return labelvec[0]
    #如果只有类别没有特征，也即是叶子节点，则投票占优类别返回。
    if len(dataset[0]) == 1 :
        return vote(dataset)
    bestFeaturePos = chooseBestFeatureToSplit(dataset)
    bestFeatureLabel = featurelabel[bestFeaturePos]
    Mytree = {bestFeatureLabel:{}}
    #featurelabel是一个列表，删掉这个特征后，后面特征对应的label会往前移。
    #但是在生成子节点的子集时，同样会把bestfeaturepos这一列的数据去掉。
    #换言之，后面的特征和featurelabel还是一一对应的。
    del(featurelabel[bestFeaturePos])
    bestFeatureVec = [example[bestFeaturePos] for example in dataset]
    bestFeatureSet = set(bestFeatureVec)
    for value in bestFeatureSet :
        #由于每一个分支之后所选取的特征不一定相同。
        #所以featurelabel去掉的特征也不同。
        #为防止冲突，这里的特征类别是值传递而不是引用传递。
        subLabels = featurelabel[:]
        Mytree[bestFeatureLabel][value] = createDecisionTree(splitDataset(dataset, bestFeaturePos, value), subLabels)
    return Mytree

def classifer(mytree, featurelabel, testvec) :
    firstlabel = list(mytree.keys())[0]
    #获取该特征的下标信息
    featurepos = featurelabel.index(firstlabel)
    #获取下一层节点的信息
    seconddict = mytree[firstlabel]
    testkey = testvec[featurepos]
    if testkey not in seconddict.keys() :
        print("The value %s of feature %s is not in this route!"%(str(testkey), str(firstlabel)))
        exit(2)
    if type(seconddict[testkey]).__name__ == 'dict' :
        return classifer(seconddict[testkey], featurelabel, testvec)
    else :
        return seconddict[testkey]


def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

if __name__=="__main__" :
    dataset, labels  = createDataSet()
    #trainlabel会被更改，所以要用复制一个集合用于训练。
    trainlabel = labels[:]
    mytree = createDecisionTree(dataset, trainlabel)
    print(mytree)
    predict = classifer(mytree, labels, [1,1])
    print(predict)
    createPlot(mytree)


