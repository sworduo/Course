#CART决策树。
#对于离散变量和连续变量的划分和处理方式不同。
#对于离散变量，必须保证在所有连续变量下都包含了所有离散变量的所有可能取值。
#对于实例中没出现的可能取值，如x1取值1,2,x2取值a，b，实例中没有（1,a）的搭配。
#这种情况下一律将（1,a）设置为此时集合的投票结果。

#此算法对于连续变量的划分只是二元划分，划分为两个子集。
#对于离散变量则是多元划分，该变量有几个取值就划分为几个子集。

#主函数流程：
#1、判断所有标签列是否一样。
#2、判断是否只剩下标签列。
#3、选择最后的用于划分集合的变量及其取值。
#4、构建树并且划分子集。
#5、对于在该集合下没有出现的某个离散变量的具体取值，一律将其标签设置为此时集合的投票。
#6、剪枝，获取基于数据集的次优决策树。

import re
from plotDT import *
import copy
#==========================================================================
#生成决策树
#==========================================================================
#求基尼系数。
#输入：    dataset [[]]
#输出：    giniCof 基尼指数
def calGini(dataset):
    labelcounts = {}
    for item in dataset :
        currentlabel = item[-1]
        if currentlabel not in labelcounts :
            labelcounts[currentlabel] = 0
        labelcounts[currentlabel] += 1
    giniCof = 1.0
    datasize = len(dataset)
    for key in labelcounts.keys() :
        pro = float(labelcounts[key]) / datasize
        giniCof -= pro * pro
    return giniCof

#对于连续变量的划分子集。
#输入：      dataset [[]]
#           axis    第几个变量
#           value   用于划分的具体变量取值
#           direction   用于指明划分的方向，0表示大于value的左子集，1表示小于等于value的右子集。
def splitContinuousDataset(dataset, axis, value, direction) :
    retDataSet = []
    for item in dataset :
        if direction == 0 :
            if item[axis] > value :
                reducedVec = item[:axis]
                reducedVec.extend(item[axis+1:])
                retDataSet.append(reducedVec)
        else :
            if item[axis] <= value :
                reducedVec = item[:axis]
                reducedVec.extend(item[axis + 1:])
                retDataSet.append(reducedVec)
    return retDataSet

#对于离散变量的划分。
#输入:    dataset [[]]
#       axis    划分变量
#       value   划分变量的具体取值
def splitDiscreteData(dataset, axis, value) :
    retDataSet = []
    for item in dataset :
        if item[axis] == value :
            reducedVec = item[:axis]
            reducedVec.extend(item[axis+1:])
            retDataSet.append(reducedVec)
    return retDataSet

#投票函数。
#输入：    classlist   []
#输出：    占优势的类别
def vote(classlist) :
    labelcounts = {}
    for label in classlist :
        if label not in labelcounts :
            labelcounts[label] = 0
        labelcounts[label] += 1
    sortedlabel = sorted(labelcounts.items(), key=lambda item : item[1], reverse=True)
    return sortedlabel[0][0]

#选择用于划分的最好的变量。
#输入:    dataset [[]]
#         labels
#输出:    bestFeature
#本来是想返回这个最佳切分变量的最佳切分点，然后主函数维护一个字典，保存每个切分变量的切分点
#然而这是不可行的。因为同一个变量在不同决策路径上的切分点是不一样的，只维护一个字典会造成冲突。
#最好的做法是，更改节点的标志，在特征标志那里加上该点的切分值。
def chooseBestFeature(dataset, labels) :
    numFeatures = len(dataset[0])-1
    bestGini = 1000000
    bestFeature = -1
    datasize = len(dataset)
    #记录每个特征的最佳切分点
    bestSplitDict = {}
    #返回连续变量的最佳切分点
    #若返回的是string型，则代表这次的最佳变量是离散变量，否则则是连续变量
    bestSplitValue = 'str'
    for i in range(numFeatures) :
        featVec = [example[i] for example in dataset]
        #连续变量
        #有一个小瑕疵。
        #这里如果连续变量刚好是4,4,4,4,4,4那么就会重复计算。
        #但是如果你set了一下去掉重复项，又不太好，因为也许最佳点就是4呢。
        #所以可以另写个函数处理一下这个特征列表，不过我懒得写了。
        if type(dataset[0][i]).__name__ == 'float' or type(dataset[0][i]).__name__ == 'int' :
            #n个取值产生n-1个切分点
            featList = sorted(featVec)
            thisFeatureGini = 10000
            for j in range(len(featList)-1) :
                thisValueGini = 0.0
                thisSplitPoint = (featList[j] + featList[j+1]) / 2
                leftSplitList = splitContinuousDataset(dataset, i, thisSplitPoint, 0)
                thisValueGini += float(len(leftSplitList)) / datasize * calGini(leftSplitList)
                rightSplitList = splitContinuousDataset(dataset, i, thisSplitPoint, 1)
                thisValueGini += float(len(rightSplitList)) / dataset * calGini(rightSplitList)
                if thisValueGini < thisFeatureGini :
                    thisFeatureGini = thisValueGini
                    bestsplitPoint = thisSplitPoint
            bestSplitDict[labels[i]] = bestsplitPoint
        # 离散变量
        else :
            thisFeatureGini = 0
            featSet = set(featVec)
            for featValue in featSet:
                thisValueList = splitDiscreteData(dataset, i, featValue)
                pro = float(len(thisValueList)) / datasize
                thisFeatureGini += pro * calGini(thisValueList)
        if thisFeatureGini < bestGini :
            bestGini = thisFeatureGini
            bestFeature = i
    #若为连续变量，返回相应切分点。
    #改变最佳切分变量的标志，指明在该点处的切分值是什么。（
    #同一个变量在不同决策路径上的切分值是不同的，所以我们需要将切分值记录在节点信息上）
    #更改该变量的取值原因是，这里只返回了切分特征，没返回具体的切分值
    #将数据集改为0-1二元值后，就可以直接调用splitDiscreteData函数了
    if type(dataset[0][i]).__name__ == 'float' or type(dataset[0][i]).__name__ == 'int' :
        bestSplitValue = bestSplitDict[labels[bestFeature]]
        labels[bestFeature] = labels[bestFeature] + "<=" + str(bestSplitValue)
        for item in dataset :
            if item[bestFeature] <= bestSplitValue :
                item[bestFeature] = 1
            else :
                item[bestFeature] = 0
    return bestFeature

#建立决策树。
#输入：    dataset [[]]
#       labels  每个特征所对应的名字。
#       data_full   完整的数据集
#       label_full  完整的名字集
#随着层数的加深，每一层节点dataset的大小在逐渐减小，出现某个变量的取值j不在该子集内部的情况。
#而对于离散变量来说，我们需要完整的数据集来计算该离散变量所拥有的所有取值的个数。
#我们需要完整的label_full来定位这一离散特征属于完整数据集里的第几列。
#输出：    myTree  决策树模型
def createDicisionTree(dataset, labels, data_full, label_full) :
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist) :
        return classlist[0]
    if len(dataset[0]) == 1 :
        return vote(classlist)
    bestFeature = chooseBestFeature(dataset, labels)
    bestFeatLabel = labels[bestFeature]
    mytree = {bestFeatLabel : {}}
    #如果特征是离散，那么需要去完整数据集里面找到这个特征的所有取值
    if type(dataset[0][bestFeature]).__name__ == 'str' :
        featPos = label_full.index(bestFeatLabel)
        fullFeatList = [example[featPos] for example in data_full]
        #该特征在完整数据集下的取值
        fullUniFeatList = set(fullFeatList)
        #该特征在当前路径下的取值
    featList = [example[bestFeature] for example in dataset]
    #此时就算是连续变量也已经变为0-1离散变量
    uniFeatList = set(featList)
    #删除这个特征
    del labels[bestFeature]
    for value in uniFeatList :
        #每条路径以后所选取的特征不同，所以需要值传递而不是引用传递
        sublabels = labels[:]
        #如果是离散特征，那么我们需要获得该特征在此路径下没有体现出来的取值。
        #方法是在完备特征值集里去掉当前路径下的取值。
        if type(dataset[0][bestFeature]).__name__ == 'str' :
            fullUniFeatList.remove(value)
        mytree[bestFeatLabel][value] = createDicisionTree(splitDiscreteData(dataset, bestFeature, value),\
                                                        sublabels, data_full, label_full)
    if type(dataset[0][bestFeature]).__name__ == 'str' :
        for value in fullUniFeatList :
            mytree[bestFeatLabel][value] = vote(classlist)

    return mytree

#预测函数。
#输入 ：   mytree 决策树模型
#输出 ：   classlabel 类别
def classify(mytree, labels, testVec) :
    firstStr = list(mytree.keys())[0]
    secondDict = mytree[firstStr]
    #如果是离散变量，需要提取变量名和切分值
    if '<=' in firstStr :
        #正则表达式：.表示匹配任意字符，+表示匹配前一个字符1到n次，‘.+’表示匹配任何字符串
        #"<=.+"表示匹配以<=开头的任意字符串
        #".+<="表示匹配以<=结尾的任意字符串
        #group()获取匹配后的字符串，是个[]
        featKey = re.compile("<=.+").search(firstStr).group()[2:]
        featLabel = re.compile(".+<=").search(firstStr).group()[:-2]
        featIndex = labels.index(featLabel)
        #小于往右子数走
        if testVec[featIndex] <= featKey :
            direction = 1
        else :
            direction = 0
        for key in secondDict.keys() :
            if key == direction :
                if type(secondDict[key]).__name__ == 'dict' :
                    return classify(secondDict[key], labels, testVec)
                else :
                    return secondDict[key]
    else :
        featIndex = labels.index(firstStr)
        for key in secondDict.keys() :
            if testVec[featIndex] == key :
                if type(secondDict[key]).__name__ == 'dict' :
                    return classify(secondDict[key], labels, testVec)
                else :
                    return secondDict[key]

#==========================================================================
#决策树剪枝
#==========================================================================
#计算子模型（即从该节点起的决策树模型分支）的错误率
def testModeError(mytree, dataset, labels) :
    error = 0.0
    for item in dataset :
        predict = classify(mytree, labels, item[:-1])
        print(predict)
        if predict != item[-1] :
            error += 1.0
    return float(error)

#计算该节点直接投票的误差
def testNodeVote(dataset, bestLabel) :
    error = 0.0
    for item in dataset :
        if item[-1] != bestLabel :
            error += 1
    return float(error)

#剪枝
#data_test是用来判断是否剪枝的测试集。
#若剪枝后的树的误差小于直接投票的误差，那么返回剪枝后的树。
#这里传进来dataset只是为了求该节点处的投票结果，然后再与测试集对比，求出直接投票的情况下，测试集的误差。
def postPruningTree(mytree, dataset, data_test, labels) :
    firstStr = list(mytree.keys())[0]
    secondDict = mytree[firstStr]
    featkey = copy.deepcopy(firstStr)
    classList = [example[-1] for example in dataset]
    if "<=" in featkey :
        featkey = re.compile(".+<=").search(featkey).group()[:-2]
        featVal = re.compile("<=.+").search(featkey).group()[2:]
    featIndex = labels.index(featkey)
    temp_labels = copy.deepcopy(labels)
    #当前节点访问过的特征需要删掉！！！为了和后面的对应
    #######
    ########
    #########
    ##########
    ###########
    #这里一定要记住删掉这一层的特征类别再传递到下一层，否则会出错。
    #因为后面的决策子树是基于这个特征类别被删掉而生成的！！！
    del labels[featIndex]
    #叶子节点不需要剪枝
    for key in secondDict.keys() :
        if type(secondDict[key]).__name__ == 'dict' :
            subLabels = labels[:]
            if type(dataset[0][featIndex]).__name__ == 'str' :
                secondDict[key] = postPruningTree(secondDict[key], splitDiscreteData(dataset, featIndex, key),\
                                                   splitDiscreteData(data_test, featIndex, key), \
                                                   subLabels)
            else :
                secondDict[key] = postPruningTree(secondDict[key], splitContinuousDataset(dataset, featIndex, featVal, key),\
                                                   splitContinuousDataset(data_test, featIndex, featVal, key), subLabels)
    majorityLabel = vote(classList)
    if testModeError(mytree, data_test, temp_labels) < testNodeVote(data_test, majorityLabel) :
        return mytree
    return majorityLabel





#==========================================================================
#主函数
#==========================================================================
if __name__ == '__main__' :
    dataset = [['dark_green', 'curl_up', 'little_heavily', 'distinct', 'sinking', 'hard_smooth', 1],\
               ['black', 'curl_up', 'heavily', 'distinct', 'sinking', 'hard_smooth', 1],\
               ['black', 'curl_up', 'little_heavily', 'distinct', 'sinking', 'hard_smooth', 1],\
               ['dark_green', 'little_curl_up', 'little_heavily', 'distinct', 'little_sinking', 'soft_stick', 1],\
               ['black', 'little_curl_up', 'little_heavily', 'little_blur', 'little_sinking', 'soft_stick', 1], \
               ['dark_green', 'stiff', 'clear', 'distinct', 'even', 'soft_stick', 0],\
               ['light_white', 'little_curl_up', 'heavily', 'little_blur', 'sinking', 'hard_smooth', 0],\
               ['black', 'little_curl_up', 'little_heavily', 'distinct', 'little_sinking', 'soft_sick', 0],\
               ['light_white', 'curl_up', 'little_heavily', 'blur', 'even', 'hard_smooth', 0],\
               ['light_white', 'stiff', 'clear', 'blur', 'even', 'hard_smooth', 0],\
               ['light_white', 'curl_up', 'little_heavily', 'blur', 'even', 'soft_stick', 0], \
               ['dark_green', 'curl_up', 'heavily', 'distinct', 'sinking', 'hard_smooth', 0], \
               ['dark_green', 'curl_up', 'heavily', 'distinct', 'sinking', 'hard_smooth', 1], \
               ['light_white', 'curl_up', 'little_heavily', 'distinct', 'sinking', 'hard_smooth', 1], \
               ['black', 'little_curl_up', 'little_heavily', 'distinct', 'little_sinking', 'hard_smooth', 1], \
               ['black', 'little_curl_up', 'heavily', 'little_blur', 'little_sinking', 'hard_smooth', 0], \
               ['dark_green', 'little_curl_up', 'little_heavily', 'little_blur', 'sinking', 'hard_smooth', 0]]
    labels = ['color', 'root', 'knocks', 'texture', 'navel', 'touch']
    label_full = labels[:]
    data_train = dataset[:14]
    data_full = data_train[:]
    mytree = createDicisionTree(data_train, labels[:], data_full, label_full)
    createPlot(mytree)
    #剪枝的第二个参数最好和上面的data_train一样。
    mytree2 = postPruningTree(mytree, dataset[:14], dataset[14:], labels)
    createPlot(mytree2)
