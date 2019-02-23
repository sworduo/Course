import numpy as np
import copy

#程序：粗浅的实现Adaboost
#日期：20180907五，晴
#作者：luo
#参考：https://www.cnblogs.com/90zeng/p/adaboost.html#contents
#adaboost是前向分步算法，在前面弱分类器的基础上去拟合新的分类器，然后根据权重得到最后的结果。
#bagging是每次从完整数据集中抽出s个样本出来，对子样本集拟合一个模型，重复n次。然后对所有n个模型求平均值（每个模型的权重都是一样的）。这里就没实现bagging了，懒。
#1、初始权重为1/M，线性扫描选择切分特征和切分值，计算alpha1，更新每个变量的权重，构建f1(x)。
#2、y=alpha1 * f1(x),预测函数为g = sign(y)，若此时y的误差为0,则退出循环。
#3、根据新的权重选择切分特征和切分值，计算alpha2,更新权重，构建f2(x)
#4、y=alpha1 * f1(x) + alpha2 * f2(x)，计算此时的总误差，若为0或小于某个阈值，退出循环。
#5、重复3、4的过程。所以Adaboost最后构建出多少个弱分类器是未知的。
class weakClassifier :
    def __init__(self):
        self.alpha = 0 #该分类器的权重
        self.axis = 0 #该分类器的切分特征下标
        self.splitVal = 0 #该分类器的切分特征值
        self.direction = 'less' #若direction='less'则小于splitVal的实例判断为-1, 若direction='great'则大于splitVal的实例判断为-1
        self.error = 0#该分类器的误差

    def setAlpha(self, a):
        self.alpha = a

    def setAxis(self, axis):
        self.axis = axis

    def setSplitVal(self, sv):
        self.splitVal = sv

    def setDirection(self, d):
        self.direction = d

    def setError(self, e):
        self.error = e

#分类器预测
#输入：X M * N 所要预测的数据集
#     axis  所选特征下标
#     splitVal 所选特征的切分值
#     direction='less'则小于切分值的实例预测为-1,若为‘great'则大于切分值的实例为-1
#输出：classPred 预测列表
def wcClassify(X, axis, splitVal, direction) :
    numSample = np.shape(X)[0]
    #默认为1
    classPred = np.ones((numSample, 1))
    if direction == 'less' :
        classPred[X[:, axis]<splitVal] = -1.0
    elif direction == 'great' :
        classPred[X[:, axis]>splitVal] = -1.0
    else :
        raise ValueError("In function wcClassify, we don't have this direction!")
    return classPred


#选择最好的切分变量
#输入:X numpyArray M × N   M是数据集大小，N是特征大小
#    y M * 1
#    D M * 1每个实例的权重
#输出：bestWC  寻找到的最好特征的弱分类器
#     minError  该特征对应的误差，用于计算alpha
#     bestClassPred 该分类器对所有实例的预测类别列表
def chooseBestFeature(X, y, D) :
    bestWC = weakClassifier()
    bestWC.setError(100000)
    numSample, numFeature = np.shape(X)
    bestClassPred = np.zeros((numSample, 1))
    numStep = 10  #对于每一个特征值，产生numStep+1个切分值
    for feat in range(numFeature) :
        rangeMin = min(X[:, feat])
        #步长四舍五入为一个整数
        step = round(((max(X[:, feat]) - rangeMin)) / numStep)
        rangeMin -= 0.5
        for i in range(0, numStep+1) :
            splitVal = rangeMin + float(i) * step
            for dir in ('less', 'great') :
                thisClassPred = wcClassify(X, feat, splitVal, dir)
                errorVec = np.ones((numSample, 1))
                errorVec[thisClassPred == y] = 0
                thisError = D.T.dot(errorVec)
                if thisError < bestWC.error :
                    bestWC.setAxis(feat)
                    bestWC.setDirection(dir)
                    bestWC.setSplitVal(splitVal)
                    bestWC.setError(thisError)
                    #这里一定要copy，因为是引用，否则thisClassPred变了，bestCP也会变？。。不确定。不过加copy肯定没问题
                    bestClassPred = copy.deepcopy(thisClassPred)
    return bestWC, bestClassPred


#训练Adaboost
#输入：    X numpy M * N
#         y numpy M * 1
#         maxIter:最大迭代次数
#输出：    adaModel    [] 里面每一个都是weakClassfier对象
def trainAda(X, y, maxIter) :
    adaModel = []
    numSample = np.shape(X)[0]
    D = np.ones((numSample, 1)) / numSample #每个实例的权重
    totalPred = np.zeros((numSample, 1)) #前n个分类器对某实例的加权总预测值
    for iter in range(maxIter) :
        thisClassifer, thisClassPred = chooseBestFeature(X, y, D)
        thisClassifer.alpha = float(np.log((1-thisClassifer.error)/thisClassifer.error)) / 2.0
        adaModel.append(thisClassifer)
        print("iterater time : %3d, alpha : %6f"%(iter+1, thisClassifer.alpha))
        print("split feature index : %3d, split feature value : %5f"%(thisClassifer.axis, thisClassifer.splitVal))
        #更新权值
        expon = np.exp(-1.0 * thisClassifer.alpha * np.multiply(y, thisClassPred))
        D = np.multiply(D, expon)
        D = D / np.sum(D)
        totalPred += thisClassifer.alpha * thisClassPred
        #如果累计分类器预测完全正确则返回
        if np.sum(np.sign(totalPred) != y) == 0 :
            break
    print("training finish!")
    return adaModel

#预测函数
#输入：    Xtext M * N
#         adaModel
#输出：    classPred 对Xtest的预测向量
def AdaClassify(Xtest, adaModel) :
    numSample = np.shape(Xtest)[0]
    classPred = np.zeros((numSample, 1))
    #遍历所有分类器
    for i in range(len(adaModel)) :
        model = adaModel[i]
        labelPred = wcClassify(Xtest, model.axis, model.splitVal, model.direction)
        classPred += labelPred * model.alpha
    return np.sign(classPred)


if __name__ == "__main__" :
    dataset = []
    for i in range(10):
        dataset.append([i])
    dataset = np.array(dataset)
    y = np.array([[1],[1],[1],[-1],[-1],[-1],[1],[1],[1],[-1]])
    model = trainAda(dataset, y, 50)
    ret = AdaClassify(np.array([[6],[5]]), model)
    print(ret)



