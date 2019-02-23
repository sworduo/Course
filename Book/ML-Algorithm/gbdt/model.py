#程序：实现gbdt
#日期：20180908-六-晴
#作者：luo
#原理参考1：https://zhuanlan.zhihu.com/p/29765582
#原理参考2：http://www.52cs.org/?p=429	
#代码参考：https://github.com/liudragonfly/GBDT
#gbdt原理参考：https://github.com/liudragonfly/GBDT/blob/master/GBDT.ipynb
from datetime import datetime
import sys
sys.path.append('~/Documents/ML-Algorithm/gbdt')
import abc
from random import  sample
from math import exp, log
from tree import *
from data import *

class RegressionLossFunction(metaclass=abc.ABCMeta) :
    def __init__(self, n_classed):
        self.K = n_classed

    #类似虚函数
    @abc.abstractmethod
    def compute_residual(self, dataset, subset, f):
        """计算残差"""

    @abc.abstractmethod
    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        """更新F的值"""

    @abc.abstractmethod
    def initialize(self, f, dataset):
        """初始化F的值"""

    @abc.abstractmethod
    def update_terminal_regions(self, y, idset):
        """更新叶子节点的预测值"""

class LeastSquareError(RegressionLossFunction) :
    """用于回归的最小平方误差函数"""
    def __init__(self, n_classes) :
        if n_classes != 1 :
            raise ValueError("n_classes must be 1 for regression but was %r"%n_classes)
        super(LeastSquareError, self).__init__(n_classes)

    #此时残差恰好等于负梯度
    def compute_residual(self, dataset, subset, f):
        residual = {}
        for id in subset :
            y = dataset.get_instance(id)['label']
            residual[id] = y - f[id]
        return residual

    #subset是本次用于训练树的样本
    #dataset是总数据集，用于获取所有的id
    #tree是这次训练的模型
    #learn_rate是和负梯度相关的学习速率，代表“前进”的距离，会一起训练
    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        data_idset = set(dataset.get_instance_idset())
        subset = set(subset)
        for node in leaf_nodes :
            for id in node.get_idset() :
                f[id] += learn_rate * node.get_predict_value()
            for id in data_idset - subset :
                f[id] += learn_rate * tree.get_predict_value(dataset.get_instance(id))

    def initialize(self, f, dataset):
        ids = dataset.get_instance_idset()
        for id in ids :
            f[id] = 0.0

    def update_terminal_regions(self, y, idset):
        sum1 = sum([y[id] for id in idset])
        return sum1 / len(idset)


class ClassificationLossFunction(metaclass=abc.ABCMeta) :
    """分类损失函数基类"""
    def __init__(self, n_classes):
        self.K = n_classes

    @abc.abstractmethod
    def compute_residual(self, dataset, subset, f):
        """计算残差"""

    @abc.abstractmethod
    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        """更新F值"""

    @abc.abstractmethod
    def initialize(self, f, dataset):
        """初始化F的值"""

    @abc.abstractmethod
    def update_terminal_regions(self, y, idset):
        """更新叶子节点的预测值"""

class BinomialDeviance(ClassificationLossFunction) :
    """二元分类的损失函数"""
    def __init__(self, n_classes):
        if n_classes != 2 :
            raise ValueError("{0:s} requires 2 classes.".format(self.__class__.__name__))
        super(BinomialDeviance, self).__init__(1)

    def compute_residual(self, dataset, subset, f):
        residual = {}
        #计算二元分类的负梯度，详见gbdt的推导,这里用的不是交叉熵损失函数
        for id in subset :
            y = dataset.get_instance(id)['label']
            residual[id] = 2.0 * y / (1 + exp(2 * y * f[id]))
        return residual

    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        data_idset = set(dataset.get_instances_idset())
        subset = set(subset)
        for node in leaf_nodes :
            for id in node.get_idset() :
                f[id] += learn_rate * node.get_predict_value()
        for id in data_idset - subset :
            f[id] += learn_rate * tree.get_predict_value(dataset.get_instance(id))

    def initialize(self, f, dataset):
        ids = dataset.get_instances_idset()
        for id in ids :
            f[id] = 0.0

    #根据最大似然估计可以得到该叶子节点的最理想的预测值
    def update_terminal_regions(self, y, idset):
        sum1 = sum([y[id] for id in idset])
        if sum1 == 0 :
            return sum1
        sum2 = sum([abs(y[id]) * (2 - abs(y[id])) for id in idset])
        return sum1 / sum2

class MultinomialDeviance(ClassificationLossFunction) :
    """多元分类的损失函数"""
    def __init__(self, n_classes, labelset):
        self.labelset = set([label for label in labelset])
        if n_classes < 3 :
            raise ValueError("{0:s} requires more than 2 classes.".format(self.__class__.__name__))
        super(MultinomialDeviance, self).__init__(n_classes)

    #softmax损失函数的负梯度=y-p
    #对于多元分类，每一次会训练m棵树，其中m=类别数量
    def compute_residual(self, dataset, subset, f):
        residual = {}
        label_valueset = dataset.get_label_valueSet()
        for id in subset :
            residual[id] = {}
            p_sum = sum([exp(f[id][x]) for x in label_valueset])
            for label in label_valueset :
                p = exp(f[id][label]) / p_sum
                y = 0.0
                if dataset.get_instance(id)["label"] == label :
                    y = 1.0
                residual[id][label] = y-p
        return residual

    #每次只更新一种类别的f值
    #每次每棵树更新完成后，只会更新这棵树所对应的label值
    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        data_idset = set(dataset.get_instance_idset())
        subset = set(subset)
        for node in leaf_nodes :
            for id in node.get_idset() :
                f[id][label] += learn_rate * node.get_predict_value()
            for id in data_idset - subset :
                f[id][label] += learn_rate * tree.get_predict_value(dataset.get_instance(id))

    def initialize(self, f, dataset):
        ids = dataset.get_instance_idset()
        for id in ids :
            f[id] = dict()
            for label in dataset.get_label_valueSet():
                f[id][label] = 0.0

    #对于每一次每一个类别的树，可能会有多个叶子节点，每个叶子节点表示在当前特征下，取这个类别的概率。
    def update_terminal_regions(self, y, idset):
        sum1 = sum([y[id] for id in idset])
        if sum1 == 0 :
            return sum1
        sum2 = sum([abs(y[id]) * (1-abs(y[id])) for id in idset])
        return ((self.K-1)/self.K) * (sum1/sum2)

class GBDT :
    #split_points不为0,并且小于数据集，那么每次只会在数据集里随机抽取split_points个样本进行训练
    def __init__(self, max_iter, sample_rate, learn_rate, max_depth, loss_type='multi-classification', split_points=0):
        self.max_iter = max_iter
        self.sample_rate = sample_rate
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.loss_type = loss_type
        self.split_points = split_points
        self.loss = None
        self.trees = dict()

    #dataset用于获取整个样本集更新所有的f值
    #train_data是实际用于训练的样本集
    def fit(self, dataset, train_data):
        #多元分类，每一次迭代训练len(classes)棵树
        if self.loss_type == 'multi_classification' :
            label_valueset = dataset.get_label_valueSet()
            self.loss = MultinomialDeviance(dataset.get_label_size(), label_valueset)
            f = dict()
            self.loss.initialize(f, dataset)
            for iter in range(1, 1+self.max_iter):
                subset = train_data
                if 0 < self.sample_rate < 1 :
                    subset = sample(subset, int(len(subset))*self.sample_rate)
                self.tree[iter] = dict()
                residual = self.loss.compute_residual(dataset, subset, f)
                for label in label_valueset :
                    #存放的是这一次迭代，这一棵树，这个类别的叶子节点
                    leaf_nodes = []
                    y = []
                    #tree训练的y是一维的，所以这里需要提取同一个label的残差值以训练属于这个label的树
                    #因为实际训练的样本只有subset，所以只需要提取subset对应的残差值即可
                    for id in subset :
                        y[id] = residual[id][label]
                    #训练这一次迭代，某一个类别的决策树
                    tree = construct_decision_tree(dataset, subset, y, 0, leaf_nodes,self.max_depth, self.loss, self.split_points)
                    self.trees[iter][label] = tree
                    self.loss.update_f_value(f, tree, leaf_nodes, subset, dataset, self.learn_rate, label)
                train_loss = self.compute_loss(dataset, train_data, f)
                print("iter: %4d : average train_loss = %6f"%(iter, train_loss))
        else :
            if self.loss_type == 'binary-classification' :
                self.loss = BinomialDeviance(n_classes=dataset.get_label_size())
            elif self.loss_type == 'regression' :
                self.loss = LeastSquareError(n_classes=1)
            else :
                raise ValueError("Loss type %s is not supported in this program!"%(self.loss_type))

            f = dict()
            self.loss.initialize(f, dataset)
            for iter in range(1, self.max_iter+1) :
                subset = train_data
                if 0 < self.sample_rate <1 :
                    subset = sample(subset, int(len(subset)*self.sample_rate))
                residual = self.loss.compute_residual(dataset, subset, f)
                leaf_nodes = []
                y = residual
                tree = construct_decision_tree(dataset, subset, y, 0, leaf_nodes, self.max_depth, self.loss, self.split_points)
                self.trees[iter] = tree
                self.loss.update_f_value(f, tree, leaf_nodes, subset, dataset, self.learn_rate)
                if isinstance(self.loss, RegressionLossFunction) :
                    #todo
                    pass
                else :
                    train_loss = self.compute_loss(dataset, train_data, f)
                    print("iter: %4d : train loss=%6f"%(iter, train_loss))

    def compute_loss(self, dataset, subset, f):
        loss = 0.0
        if self.loss.K == 1 :
            for id in dataset.get_instances_idset() :
                y = dataset.get_instance(id)['label']
                f_value = f[id]
                p_1 = 1/(1+exp(-2*f_value))
                try:
                    loss -= ((1+y)*log(p_1)/2) + ((1-y)*log(1-p_1)/2)
                except ValueError as e :
                    print(y, p_1)
        else :
            for id in dataset.get_instance_idset() :
                instance = dataset.get_instance(id)
                f_values = f[id]
                exp_values = {}
                for label in f_values :
                    exp_values[label] = exp(f_values[label])
                probs = {}
                for label in f_values :
                    probs[label] = exp_values[label] / sum(exp_values.values())
                loss -= log(probs[instance["label"]])
        return loss / dataset.size()

    def compute_instance_f_value(self, instance) :
        if self.loss.K == 1 :
            f_value = 0.0
            for tree in self.trees :
                f_value += self.learn_rate * tree.get_predict_value(instance)
        else :
            f_value = dict()
            for label in self.loss.labelset :
                f_value[label] = 0.0
            for iter in self.trees :
                for label in self.loss.labelset :
                    tree = self.trees[iter][label]
                    f_value[label] += self.learn_rate * tree.get_predict_value(instance)
        return f_value

    def predict(self, instance):
        """回归和二元分类返回f值，多元分类返回每一类的f值"""
        return self.compute_instance_f_value(instance)

    def predict_prob(self, instance):
        """返回每个类别的概率"""
        if isinstance(self.loss, RegressionLossFunction) :
            raise RuntimeError('regression problem can not predict prob!')
        if self.K == 1 :
            f_value = self.compute_instance_f_value(instance)
            probs = dict()
            probs['+1'] = 1/(1+exp(-2*f_value))
            probs['-1'] = 1 - probs['+1']
        else :
            f_value = self.compute_instance_f_value(instance)
            exp_values = dict()
            for label in f_value :
                exp_values[label] = exp(f_value[label])
            exp_sum = sum(exp_values.values())
            probs = dict()
            for label in exp_values :
                probs[label] = exp_values[label] / exp_sum
        return probs

    def precidt_label(self, instance):
        """预测标签"""
        predict_label = None
        if isinstance(self.loss, BinomialDeviance) :
            probs = self.predict_prob(instance)
            predict_label = 1 if probs['+1'] >= probs['-1'] else -1
        else :
            probs = self.predict_prob(instance)
            for label in probs :
                if not predict_label or probs[label] > probs[predict_label]:
                    predict_label = label
        return predict_label























#网上现在的gbdt几乎都是在讲xgboost...下面应该算是xgboost的思路。
#gbdt的核心问题是，在前n-1个模型yn-1 确定的情况下，怎样选取第n个模型（通常是决策树）的决策规则，
#使得总误差最少。也就是opt = 连加(L(y, yn)+正则项)最小。
#正则项是叶子节点数量加上每个叶子节点的权值向量的二范数平方。
#和线性回归类似，wj(第j个叶子节点的输出)是我们所需要求的参数，所以可以加上其二范数作为正则项防止过拟合。
#gbdt优化的方法，就是将L(y,yn)=L(y,yn-1+fn)在yn-1处泰勒展开。
#然后合并化简。将对每个数据的最优化转换为对于每个叶子节点的最优化。
#由于opt最后是关于w的函数（我们要求w），所以只需要最后对w求导=0得到opt最优化时w的取值。
#回带到原来的opt式子，得到opt在当前w下的最优化值，也就是当前数据集下，所能选取的最好w所对应的误差。
#所以建树的时候，可以根据误差的差值来选取相应的变量。
#每一次都要计算L关于yn-1的一阶导和二阶导，以此来求得新树的w。
#L对yn-1求导将会得到y和yn-1的关系式，每次更新yn-1后套尽关系式里就可以得到一阶导的值了。
#就好象L=0.5*(y-yn-1)^2一样，负梯度是y-yn-1，那么每次计算出新的yn-1值，再套尽表达式里就得到负梯度了。


#还有一个问题就是，第n个模型该选择什么数据来进行拟合。
#直接拟合L对前n-1个模型yn-1的负梯度进行拟合就可以了。
#可以这么来理解：目标函数L是关于前n-1个模型yn-1的的函数。
#L对yn-1的负梯度指明了yn-1的更新方向。
#(yn=yn-1+L对其的负梯度)则完成了一次参数更新（就是梯度下降的原理，L(yn-1+负梯度)会比L(yn-1)更接近L的极值点)
#当然，就好象梯度下降除了梯度外，还有学习速率（也就是步长），gbdt同样也要求得最优的步长值。
#当损失函数是均方误差的时候，负梯度恰好等于其残差。
