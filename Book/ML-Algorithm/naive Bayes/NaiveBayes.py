import numpy as np


#用于文本处理的朴素贝叶斯，参考https://blog.csdn.net/Dream_angel_Z/article/details/46120867
# 1、加载数据
# 2、构建词库
# 3、将数据转换为词向量
# 4、用词向量计算先验概率
# 5、用先验概率进行预测
class NaiveBayes:
    def loadData(self):
        rawLine = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        #0为好的东西，1是不好的东西。
        label = np.array([0, 1, 0, 1, 0, 1])
        return rawLine, label

    #创建词库
    def createLib(self, data):
        wordLib = set([])
        for line in data :
            wordLib = wordLib | set(line)
        return list(wordLib)

    #转换一行词向量
    #因为预测的时候只有一行，而且也要进行词向量转换
    #所以这里要写成一行转换，而不是整个矩阵转换的形式。
    def transformWord(self, inputLine, wordLib):
        returnVec = [0]*len(wordLib)
        for word in inputLine :
            if word in wordLib :
                returnVec[wordLib.index(word)] = 1
            else :
                print("The word %s is not in my vocabulary!"%word)
        return returnVec

    #计算条件概率和类别概率
    #vecMatrix和label都是list
    def calProbability(self, vecMatrix, label):
        #label只有0和1,所以求和就是1的个数
        P1 = sum(label)/float(len(label))
        if len(vecMatrix) == 0 :
            print("The vecMatrix is empty")
            exit(1)
        n = len(vecMatrix[0])
        #拉普拉斯平滑，每个维度个数首先设为1,防止出现条件概率为0的情况
        P0num = np.ones(n)
        P1num = np.ones(n)
        #由于每个实例中每个维度只有0和1两种可能，所以分母预先设置为2
        P0denom = 2.0
        P1denom = 2.0
        for i in range(len(label)) :
            if label[i] == 1 :
                P1num += vecMatrix[i]
                P1denom += 1.0
            else :
                P0num += vecMatrix[i]
                P0denom += 1.0
        prob0 = P0num/P0denom
        prob1 = P1num/P1denom
        return prob0, prob1, P1

    #预测函数
    def predict(self, vec, prob0, prob1, P1):
        #这里是对求总的条件概率的公式求了个log，防止因为太多小数相乘导致下溢
        #np.ones*[]得到的是一个列表，相当于对应元素相乘
        score0 = sum(vec*np.log(prob0)) + np.log(1.0-P1)
        score1 = sum(vec*np.log(prob1)) + np.log(P1)
        #log是单调函数，log后更大表明原来的也更大。
        if score0 > score1 :
            return 0
        else :
            return 1


    def train(self):
        rawLine, label = self.loadData()
        mylib = self.createLib(rawLine)
        vecMatrix = []
        for line in rawLine :
            vecMatrix.append(self.transformWord(line, mylib))
        prob0, prob1, P1 = self.calProbability(vecMatrix, label)
        #测试
        testline = ['love', 'my', 'dalmation']
        testvec = self.transformWord(testline, mylib)
        print(testline, "classified as: ", self.predict(testvec, prob0, prob1, P1))
        # 测试2
        testline = ['stupid', 'garbage']
        testvec = self.transformWord(testline, mylib)
        print(testline, "classified as: ", self.predict(testvec, prob0, prob1, P1))


if __name__=="__main__" :
    nb = NaiveBayes()
    nb.train()
