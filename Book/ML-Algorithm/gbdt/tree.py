from math import log
from  random import sample


#create decision tree
#author: luo
#date: 20190913 Thu. sun

#普通的树节点
class Tree :
    def __init__(self):
        self.split_feature = None
        self.leftTree = None
        self.rightTree = None
        #对于实数类型的判别条件为<conditionVal,对于str类型为=
        #将满足条件的数据放入左树
        self.real_value_feature = True
        self.conditionVal = None
        self.leafNode = None

    def get_predict_value(self, instance):
        #是叶子节点
        if self.leafNode :
            return self.leafNode.get_predict_value()
        if not self.split_feature :
            raise ValueError("the tree is null")
        if self.real_value_feature and instance[self.split_feature] < self.conditionVal :
            return self.leftTree.get_predict_value(instance)
        elif not self.real_value_feature and instance[self.split_feature] == self.conditionVal :
            return self.leftTree.get_predict_value(instance)
        return self.rightTree.get_predict_value(instance)

    def describe(self, addtion_info="") :
        if not self.leftTree or not self.rightTree :
            return self.leafNode.describe()
        leftInfo = self.leftTree.describe()
        rightInfo = self.rightTree.describe()
        info = addtion_info+"{split_feature: "+str(self.split_feature)+", splie_value: "+str(self.conditionVal)+\
               "[left_tree "+leftInfo+", right_tree:"+rightInfo+"]}"
        return info

#叶子节点
class leafNode :
    def __init__(self, idset):
        self.idset = idset
        self.predictVal = None

    def describe(self):
        return "{LeafNode: "+str(self.predictVal)+" }"

    def get_idset(self):
        return self.idset

    def get_predict_value(self):
        return self.predictVal

    #y为实际类别，loss是所选择的损失函数，不同损失函数的误差计算公式不一样
    def update_predict_value(self, y, loss) :
        self.predictVal = loss.update_terminal_regions(y, self.idset)

#均方误差
def MSE(values) :
    if len(values) < 2 :
        return 0
    mean = sum(values) / float(len(values))
    error = 0.0
    for i in values :
        error += (i - mean) * (i - mean)
    return error

#参考Friedman的论文Greedy Function Approximation: A Gradient Boosting Machine中公式35
def FriedmanMSE(left_values, right_values) :
    weighted_n_left, weighted_n_right = len(left_values), len(right_values)
    total_meal_left, total_meal_right = sum(left_values) / float(weighted_n_left), sum(right_values) / float(len(right_values))
    diff = total_meal_left - total_meal_right
    return (weighted_n_left * weighted_n_right * diff * diff / (weighted_n_left + weighted_n_right))

#dataset    完整数据集
#remainedSet    用于训练的数据集样本（每次都是随机抽取一部分数据来训练）
#y  对应样本的类别
#depth 当前深度
#leaf_nodes 记录所有叶子节点
#max_depth 最大深度，用于控制树的高度
#criterion 用来判断选择哪个特征来进行切分当前节点的依据，类似于信息增益
#loss 用于叶子节点计算叶子节点的输出值。（根据gbdt的推导，不同损失函数在叶子节点的输出值不一样）
#split_points 决定每个特征用于线性扫描的样本数。 假如某特征有100个值，若split_points为50,则会随机抽取50个值来扫描决定该特征的最佳切分点。

def construct_decision_tree(dataset, remainedSet, y, depth, leaf_nodes, max_depth, loss, criterion='MSE', split_points=0) :
    if depth < max_depth :
        attributes = dataset.get_attributes()
        mse = -1
        seletedAttribute = None
        conditionValue = None
        selectedLeftIdset = []
        selectedRtightIdset = []
        for attribute in attributes :
            is_real_type = dataset.is_real_feature(attribute)
            attrValues = dataset.get_label_valueSet(attribute)
            if is_real_type and split_points > 0 and len(attrValues) > split_points :
                attrValues = sample(attrValues, split_points)
            for attrValue in attrValues :
                leftIdset = []
                rightIdset = []
                for id in remainedSet :
                    instance = dataset.get_instance(id)
                    value = instance[attribute]
                    if (is_real_type and value < attrValue) or (not is_real_type and value == attrValue) :
                        leftIdset.append(id)
                    else :
                        rightIdset.append(id)
                leftTargets = [y[id] for id in leftIdset]
                rightTargets = [y[id] for id in rightIdset]
                sum_mse = MSE(leftTargets) + MSE(rightTargets)
                if mse < 0 or sum_mse < mse :
                    seletedAttribute = attribute
                    conditionValue = attrValue
                    mse = sum_mse
                    selectedLeftIdset = leftIdset
                    selectedRtightIdset = rightIdset
        if not seletedAttribute or mse < 0 :
            raise ValueError("cannot determine the split attribute.")
        tree = Tree()
        tree.split_feature = seletedAttribute
        tree.real_value_feature = dataset.is_real_feature(seletedAttribute)
        tree.conditionVal = conditionValue
        tree.leftTree = construct_decision_tree(dataset, selectedLeftIdset, y, depth+1, leaf_nodes, max_depth, loss)
        tree.rightTree = construct_decision_tree(dataset, selectedRtightIdset, y, depth+1, leaf_nodes, max_depth, loss)
        return tree
    #叶子节点
    else :
        node = leafNode(remainedSet)
        node.update_predict_value(y, loss)
        leaf_nodes.append(node)
        tree = Tree()
        tree.leafNode = node
        return tree
