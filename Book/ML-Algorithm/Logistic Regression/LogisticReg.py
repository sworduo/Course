import numpy as np
import matplotlib.pyplot as plt
import time
import random


#logistic regression
#2018-09-05，三，六七二一
#参考代码https://blog.csdn.net/zouxy09/article/details/20319673

def Sigmoid(z) :
    return 1.0 / (1.0 + np.exp(-z))

#logistic regression训练函数
#输入：    X   np矩阵 N*m，N是数据集大小，m是特征个数，包含偏置单元1的矩阵
#         y   np矩阵 N×1 对应实例的类别
#         opts:包含了训练所需要的参数：
#                               maxIter：最大迭代次数
#                               alpha：学习速率
#                               optimalType：所选的优化方法：
#                                           GD梯度下降 SGD 随机梯度下降 smoothSGD 学习速率会随着迭代次数调整的随机梯度下降
#

#对于numpy矩阵x[i:i+1,:]返回一个只有一行的矩阵！1×n，是二维的
#x[i, :]返回一个列表，是一维的
#x[2]获取某一行，x[2,2]获取某一个数字

#X[i,:]是一维的，没有转置！！X[i:i+1, :]是一个矩阵，才有转置！
def trainLR(X, y, opts) :
    numSample, numFeature = np.shape(X)
    maxIter = opts['maxIter']
    alpha = opts['alpha']
    weights = np.ones((numFeature, 1))
    stratTime = time.time()
    for iter in range(maxIter) :
        if opts['optimalType'] == 'GD' :
            output = Sigmoid(X.dot(weights))
            error = X.T.dot(output - y)
            weights -= error * alpha
        elif opts['optimalType'] == 'SGD' :
            for i in range(numSample) :
                output = Sigmoid(X[i, :].dot(weights))
                error = X[i:i+1, :].T * (output - y[i, 0])
                weights -= alpha * error
        #随着迭代次数增加降低学习速率
        #每一次随机选择元素更新，防止数据周期抖动。
        elif opts['optimalType'] == 'smoothSGD' :
            #每一次随机梯度下降的顺序都是打乱的
            dataIndex = list(range(numSample))
            # ??很奇怪，如果不注释下面这条打乱顺序的语句
            #那么学习不到好的结果。
            #然而这只是一个随机选择训练数据下标的东西，奇怪。
            # random.shuffle(dataIndex)
            for i in range(numSample) :
                randIndex = dataIndex[i]
                alpha = 4.0 / (1.0 + iter + i) + 0.01
                output = Sigmoid(X[randIndex, :].dot(weights))
                error = X[randIndex:randIndex+1, :].T * (output - y[i, 0])
                weights -= alpha * error
        else :
            raise NameError("Not support optimize method type!")
    print("Training complete, took %s !"%(time.time()-stratTime))
    return weights

#预测函数。
#输入 weights m*1矩阵
#   Xtest n*m矩阵或者1*m矩阵
#输出 预测类别
def classify(weights, Xtest) :
    return Sigmoid(Xtest.dot(weights)) > 0.5

#画图函数，只能画2特征函数（不包括偏置单元）
def plotLR(weights, X, y) :
    numSample, numFeature = np.shape(X)
    if numFeature != 3 :
        print("Sorry! I can not drwa this picture")
        exit(1)

    for i in range(numSample) :
        if int(y[i, 0]) == 0 :
            plt.plot(X[i, 1], X[i, 2], 'or')
        else :
            plt.plot(X[i, 1], X[i, 2], 'ob')

    #直线是w0 + x1*w1 +x2*w2 = 0
    min_x1 = min(X[:, 1])
    max_x1 = max(X[:, 1])
    min_x2 = float(-weights[0, 0] - weights[1, 0] * min_x1) / weights[2, 0]
    max_x2 = float(-weights[0, 0] - weights[1, 0] * max_x1) / weights[2, 0]
    plt.plot([min_x1, max_x1], [min_x2, max_x2], '-g')
    plt.show()


if __name__ == '__main__' :
    fr = open("data.txt")
    X = []
    y = []
    for line in fr.readlines() :
        lineArr = line.strip().split()
        X.append([1, float(lineArr[0]), float(lineArr[1])])
        y.append([float(lineArr[2])])

    X = np.array(X)
    y = np.array(y)
    m, n = np.shape(X)
    print("data size:%d,  feature size :%d"%(m, n-1))
    opts = {}
    opts['alpha'] = 0.01
    opts['maxIter'] = 500
    opts['optimalType'] = 'smoothSGD'
    weights = trainLR(X, y, opts)
    plotLR(weights, X, y)





