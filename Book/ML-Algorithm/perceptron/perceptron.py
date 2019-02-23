import numpy as np


class perceptron:
    #构造函数：
    #输入：
    #X  ：特征集，n×m矩阵，其中m是特征维数
    #y  ：类别,n*1矩阵
    #lr ：学习速率
    #b  ：阈值
    #self.n :数据集维度
    #a  :代表每个实例对权重的贡献，n×1矩阵
    #self.G :Gram矩阵
    #默认假设a初始值全是0
    def __init__(self, X, y, lr = 1, b = 0):
        self.n = np.shape(X)[0]
        self.a = np.array([[0.0]] * self.n)
        self.G = X.dot(X.T)
        self.y = y
        self.b = b
        self.lr = lr
        self.cnt = 0

    def sign(self, x):
        if x > 0 :
            return 1
        return -1

    def f(self, x):
        return self.sign(np.multiply(self.a, self.y).T.dot(x)+self.b)

    #用这个算法前必须保证数据集是可分的，否则会陷入死循环
    def train(self):
        #flag判断算法是否结束
        flag = 0
        while flag == 0 :
            flag = 1
            #flag1判断是否有实例违反规则，若是，则更新后重头开始遍历。
            flag1 = 0
            for iter in range(self.n):
                while y[iter] != self.sign(self.f(self.G[:, iter:iter+1])) :
                    flag1 = 1
                    flag = 0
                    self.a[iter] += self.lr
                    self.b += self.lr * self.y[iter]
                    self.cnt += 1
                    print("the times of update : %d"%self.cnt)
                    print("error point : X%d "%(iter+1))
                    print("the value of a : ", end='')
                    for i in range(self.n-1):
                        print("%d "%self.a[i], end='')
                    print("%d"%self.a[-1])
                    print("the value of b : %d"%self.b)
                    print()
                if flag1 == 1 :
                    break


if __name__=="__main__" :
    #例子来自于《统计学习方法》P34例2.2
    X = np.array([[3,3], [4,3], [1,1]])
    y = np.array([[1], [1], [-1]])
    per = perceptron(X, y)
    per.train()






