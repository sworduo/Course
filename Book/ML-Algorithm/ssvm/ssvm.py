import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy

#程序：使用SMO算法实现SVM
#日期：20180906 四，晴
#参考：https://blog.csdn.net/zouxy09/article/details/17292011

#SMO算法：
#首先考虑二维两个变量的情况，我们一般选择固定y只更新x，然后固定x只更新y，重复这两个过程直到函数去到最小点。
#SMO算法每次更新两个算法是因为约束条件的存在，若只选择一个变量更新，那么根据约束条件这个变量就变成定值了。
#如同A->B->C->D,每次其实只有一个变量在改变，但最后能去到D这里。
#   B--------->
#   ^         |
#   |         |
#   |         v
#   |         C------>D
#   A


# 计算某一行特征和其余所有特征的核函数值
# 输入：Xmatrix M × N
#     Xvec    1 × N
# 返回  kernelVec = M * 1
def calKernelVec(Xmatrix, Xvec, kernelOpt):
    kernelType = kernelOpt[0]
    numSample = np.shape(Xmatrix)[0]
    kernelVec = np.zeros((numSample, 1))

    if kernelType == 'linear':
        kernelVec = Xmatrix.dot(Xvec.T)
    #高斯核函数 = (xi - xj ) / (-2 * sigma^2)
    elif kernelType == 'rbf':
        sigma = kernelOpt[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(numSample) :
            diff = Xmatrix[i:i+1, :] - Xvec
            kernelVec[i] = np.exp(diff.dot(diff.T) / (-2.0 * sigma**2))
    else :
        raise NameError("In function calKernelVec:%s is not in our kernelType List"%(kernelOpt[0]))
        exit(1)
    return kernelVec

#计算核函数矩阵
#X  M × N
#kernelType (kernelType, sigma)
def calKernelMatrix(X, kernelOpt):
    numSample = np.shape(X)[0]
    kernelMatrix = np.zeros((numSample, numSample))
    for i in range(numSample) :
        kernelMatrix[:, i:i+1] = calKernelVec(X, X[i:i+1, :], kernelOpt)
    return kernelMatrix

class ssvm:
    #构造函数：
    #dataset numpy矩阵 M × N M代表数据集大小，N代表特征大小
    #labels M × 1对应实例标签
    #C  松弛变量前面的系数
    #toler 阈值，当alpha的更新值小于该值时，不更新第二个变量，在改变自己的E后直接返回。
    #kernelOpt 元组 （kernelType, sigma)
    #               kernelType为'linear'时，核函数直接相乘，这时不需要第二个参数。
    #                         为'rbf'的时候，第二个参数为sigma
    def __init__(self, dataset, labels, C, toler, kernelOpt=('rbf', 1.0)):
        self.X = dataset
        self.y = labels
        self.C = C
        self.toler = toler
        self.numSample = np.shape(self.X)[0]
        self.alpha = np.zeros((self.numSample, 1))
        self.b = 0
        #errorCache记录每一个实例的预测值和实际值的差。
        #第一列表示这个alpha有没有更新误差值，0表示没有，1表示有。
        #第二列是这个误差值的具体值。
        #选择第二个变量时，是从errorCache里第一列为1的那些变量里选择。
        #因为第一列为1代表更新过，也意味着很可能是边界附近的点（因为边界附近的点对决策边界很敏感，所以可能经常需要更新，也是更新后效果最明显的）
        #或者说，想象一下在决策边界附近有10个点，远离决策边界1000000米的地方有一亿个点。
        #我们根本不需要管那一亿个点，因为决策边界的更新对他们几乎没影响。
        #或者说折一亿个点是几乎不会被选择的点，自然我们就不会去从这一亿个点中寻找更新变量。
        #这个errorCache就是用来区分那一亿个点和边界附近的点的列表。
        self.errorCache = np.zeros((self.numSample, 2))
        self.kernelOpt = kernelOpt
        self.kernelMat = calKernelMatrix(self.X, self.kernelOpt)

    #计算误差值
    def calError(self, alpha_i):
        out = np.multiply(self.alpha, self.y).T.dot(self.kernelMat[:, alpha_i:alpha_i+1]) + self.b
        return out - float(self.y[alpha_i])

    #更新误差值
    def updateError(self, alpha_i):
        error = self.calError(alpha_i)
        self.errorCache[alpha_i] = [1, error]

    #选择第二个变量j
    def selectJ(self, alpha_i, error_i):
        #i是一个需要更新的变量，所以记录它的误差值
        self.errorCache[alpha_i] = [1, error_i]
        candidateList = np.nonzero(self.errorCache[:, 0])[0]
        alpha_j = -100
        error_j = 0
        #一开始的时候，cacheError(i)已经更新了，所以只有长度大于1才证明还有其他的alpha更新过。
        if len(candidateList) > 1 :
            maxStep = 0
            for j in candidateList :
                if j == alpha_i :
                    continue
                thisError = self.calError(j)
                thisStep = abs(error_i - thisError)
                #这里写大于等于的原因：
                #考虑errorCache里只有j和alpha_i两个元素，若calerror(j)恰好等于error_i
                #那么若判断条件是大于，那么就不会更新alpha_j，换言之，alpha_j返回值就是-100。
                #当然这个问题也可以通过设置alpha_j初始值为0来解决。
                if thisStep >= maxStep :
                    maxStep = thisStep
                    error_j = thisError
                    alpha_j = j
        #刚开始的时候随机选择一个j
        else :
            alpha_j = alpha_i
            while alpha_j == alpha_i :
                #randint上下限都有可能取到
                alpha_j = random.randint(0, self.numSample-1)
            error_j = self.calError(alpha_j)
        return alpha_j, error_j


    #内循环，启发式搜索第二个变量j，然后更新。
    #返回值：1,本次有更新，0本次无更新
    def innerLoop(self, alpha_i):
        #检查是否违反KKT条件
        #y(wx+b)=1 => y(wx+b)-y*y=0 => y(wx+b-y)=0 => yE=0
        #可以通过判断yE的值来判断y(wx+b)是否大于1或小于1
        #以下列举违反KKT条件的情况：
        #a<C为支持向量和正确分类的点，理应yE>=0若此是yE<0则代表违反KKT条件
        #若a>0为支持向量和错误分类的点理应yE<=0若此是yE>0则代表微分KKT条件
        #支持向量本来是应该yE=0，但是后面sv更改后，可能会影响前面的SV的取值，也就是说，随着优化的进行，前面的sv可能不再是sv，需要重新调整。
        error_i = self.calError(alpha_i)
        if (self.y[alpha_i]*error_i < -self.toler and self.alpha[alpha_i] < self.C) or \
                    (self.y[alpha_i]*error_i > self.toler and self.alpha[alpha_i] > 0) :
            alpha_j, error_j = self.selectJ(alpha_i, error_i)
            #先记录两个alpha的旧值，因为变量是一个值，所以需要用到copy，否则单纯的赋值=只是引用
            alpha_i_old = copy.deepcopy(self.alpha[alpha_i])
            alpha_j_old = copy.deepcopy(self.alpha[alpha_j])
            #计算LH
            if self.y[alpha_j] != self.y[alpha_i] :
                L = max(0, self.alpha[alpha_j] - self.alpha[alpha_i])
                H = min(self.C, self.C + self.alpha[alpha_j] - self.alpha[alpha_i])
            else :
                L = max(0, self.alpha[alpha_j] + self.alpha[alpha_i] - self.C)
                H = min(self.C, self.alpha[alpha_j] + self.alpha[alpha_i])
            if L == H :
                return 0
            #计算两个变量的相似程度
            eta = 2.0 * self.kernelMat[alpha_i , alpha_j] - self.kernelMat[alpha_j, alpha_j] - \
                  self.kernelMat[alpha_i, alpha_i]
            if eta >= 0 :
                return 0
            self.alpha[alpha_j] -= float(self.y[alpha_j] * (error_i - error_j) / eta)
            if self.alpha[alpha_j] > H :
                self.alpha[alpha_j] = H
            if self.alpha[alpha_j] < L :
                self.alpha[alpha_j] = L
            #如果j的更新值非常小，就直接返回
            if abs(alpha_j_old - self.alpha[alpha_j]) < 0.00001 :
                self.updateError(alpha_j)
                return 0
            self.alpha[alpha_i] += self.y[alpha_i] * self.y[alpha_j] * (alpha_j_old - self.alpha[alpha_j])
            b1 = self.b - error_i - self.y[alpha_i] * (self.alpha[alpha_i] - alpha_i_old) \
                    * self.kernelMat[alpha_i, alpha_i] - self.y[alpha_j] * (self.alpha[alpha_j] - alpha_j_old) \
                    * self.kernelMat[alpha_i, alpha_j]
            b2 = self.b - error_j - self.y[alpha_i] * (self.alpha[alpha_i] - alpha_i_old) \
                    * self.kernelMat[alpha_i, alpha_j] - self.y[alpha_j] * (self.alpha[alpha_j] - alpha_j_old) \
                    * self.kernelMat[alpha_j, alpha_j]
            if (0 < self.alpha[alpha_i]) and (self.alpha[alpha_i] < self.C) :
                self.b = b1
            elif (0 < self.alpha[alpha_j]) and (self.alpha[alpha_j] < self.C) :
                self.b = b2
            else :
                self.b = (b1 + b2) / 2.0
            self.updateError(alpha_j)
            self.updateError(alpha_i)
            return 1
        else :
            return 0

    #svm训练函数
    #maxIter    最大迭代次数
    #流程：首先在数据集寻找违反KKT条件的变量，然后在支持向量集里面找。
    #支持向量集若有违反KKT条件的变量，则更新后还是要在sv集里再寻找一遍。
    #支持向量集遍历后没有违反KKT条件的变量，就返回数据集找。在这两个集合里切换。
    #一直到在整个数据集里都没有违反KKT条件的变量就退出循环。
    def trainSVM(self, maxIter):
        startTime = time.time()
        entireSet = 1 #为1则更新整个数据集，为0更新支持向量集
        alphaPairChanged = 0 #alpha更新个数
        iter = 0
        #结束条件：满足迭代次数;或者遍历整个数据集后没有alpha进行更新
        while (iter < maxIter) and (entireSet == 1 or alphaPairChanged != 0) :
            #每次更新前重置alphaPairChanged
            #若不重置则永不为0,就永远无法结束循环。
            iter += 1
            alphaPairChanged = 0
            if entireSet == 1 :
                for i in range(self.numSample) :
                    alphaPairChanged += self.innerLoop(i)
                print("---iter : %5d , entire set, alpha pair changed : %5d"%(iter, alphaPairChanged))
            else :
                SVlist = np.nonzero(np.multiply(self.alpha > 0, self.alpha < self.C))[0]
                for i in SVlist :
                    alphaPairChanged += self.innerLoop(i)
                print("---iter : %5d , support vector list, alpha pair changed : %5d" % (iter, alphaPairChanged))
            if entireSet == 1 :
                entireSet = 0
            #entireset==0表示这次更新的是支持向量集
            #若支持向量集里面没有alpha更新，那么就回到整个数据集寻找违反KKT条件的变量。
            elif alphaPairChanged == 0 :
                entireSet = 1

        print("Congratulations, training complete! Took %fs!" % (time.time() - startTime))

    #预测函数
    def testSVM(self, Xtest, ytest):
        SVlist = np.nonzero(np.multiply(self.alpha>0, self.alpha<self.C))[0]
        SVlabels = self.y[SVlist]
        SVmatrix = self.X[SVlist, :]
        testSample = np.shape(Xtest)[0]
        SValpha = self.alpha[SVlist]
        matchCnt = 0
        for i in range(testSample) :
            kernelVec = calKernelVec(SVmatrix, Xtest[i:i+1, :], self.kernelOpt)
            predict = np.multiply(SValpha, SVlabels).T.dot(kernelVec)
            if predict * ytest[i] > 0 :
                matchCnt += 1
        accuracy = float(matchCnt) / testSample
        return accuracy

if __name__ == '__main__' :
    fr = open("data.txt")
    X = []
    y = []
    for line in fr.readlines() :
        lineArr = line.strip().split()
        insertLine = [float(lineArr[0]), float(lineArr[1])]
        X.append(insertLine)
        y.append([float(lineArr[2])])
    X = np.array(X)
    y = np.array(y)
    Xtrain = X[:50, :]
    ytrain = y[:50, :]
    Xtest = X[50:, :]
    ytest = y[50:, :]
    model = ssvm(Xtrain, ytrain, 0.6, 0.001)
    model.trainSVM(500)
    accuracy = model.testSVM(Xtest, ytest)
    print("accuracy : %f"%(accuracy * 100))


