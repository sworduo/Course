#处理数据
#作者：luo
#日期：20180912三，晴、雨

#二元分类问题标签：{-1,+1}
class dataset:
    def __init__(self, filename):
        line_cnt = 0
        #记录每一行实例
        self.instances = dict()
        #统计特征类型为浮点型或者int型的特征的有多少个不同的取值
        self.distinct_valueset = dict()
        for line in open(filename):
            if line == "\n":
                continue
            #最后一个字符为换行符
            curLine = line[:-1].split(",")
            if line_cnt == 0 :
                #csv头部，记录特征的名字。
                self.feature_names = tuple(curLine)
            else :
                if len(self.feature_names) != len(curLine) :
                    print("wrong input line : ", line)
                    raise ValueError("The number of this line is not match other line!")
                if line_cnt == 1 :
                    #记录每一个特征的类型，str或者float
                    self.feature_type = dict()
                    #str型对应的feature_type集合不为0,float对应的feature_type集合为0,以此来判断一个特征是不是实数
                    for i in range (len(self.feature_names)) :
                        valueset = set()
                        try :
                            float(curLine[i])
                            self.distinct_valueset[self.feature_names[i]] = set()
                        except ValueError :
                            valueset.add(curLine[i])
                        self.feature_type[self.feature_names[i]] = valueset
                self.instances[line_cnt] = self.contruct_instance(curLine)
            line_cnt += 1
        
    def contruct_instance(self, curLine):
        instance = dict()
        for i in range(len(self.feature_names)) :
            featureType = self.feature_names[i]
            if self.is_real_feature(featureType) :
                try :
                    #读取文件时会把数值型数据转化为str型，现在要转回float
                    instance[featureType] = float(curLine[i])
                    self.distinct_valueset[featureType].add(float(curLine[i]))
                except ValueError:
                    raise ValueError("the value is not float!")

            else :
                instance[featureType] = curLine[i]
                self.feature_type[featureType].add(curLine[i])
        return instance
        
    #判断是否是实数类型特征
    def is_real_feature(self, featureType):
        if featureType not in self.feature_names :
            raise ValueError("feature name not in the dictionay of dataset")
        return len(self.feature_type[featureType]) == 0

    #获取某个特征有多少个不同的取值
    #默认是获取有多少个类别
    def get_label_size(self, featureName="label") :
        if featureName not in self.feature_names :
            raise ValueError("feature name not in the dictionay of dataset")
        #若为数值型，则第一项为0返回第二项
        return len(self.feature_type[featureName]) or len(self.distinct_valueset[featureName])

    def get_instances_idset(self) :
        return set(self.instances.keys())

    #获取某个特征的数据列表
    def get_label_valueSet(self, featureName="label") :
        if featureName not in self.feature_names :
            raise ValueError("there is no class label name!")
        return self.feature_type[featureName] if self.feature_type[featureName] else self.distinct_valueset[featureName]

    #返回样本个数
    def size(self):
        return len(self.instances)

    #根据ID获取样本
    def get_instance(self, id):
        if id not in self.instances :
            raise ValueError("Id not in the instances dict of dataset!")
        return self.instances[id]

    #返回所有feature的名称
    def get_attributes(self):
        feature_name = [x for x in self.feature_names if x != "label"]
        return tuple(feature_name)
            
    #描述数据集相关属性
    def describe(self):
        info = "feature: "+str(self.feature_names)+"\n"
        info = info + "\n dataset size="+str(self.size())+"\n"
        for feat in self.feature_names :
            info = info + "description for feature: " + feat
            valueset = self.get_label_valueSet(feat)
            if self.is_real_feature(feat) :
                info = info + "real value, distinct values number: "+str(len(valueset))
                info = info + " range is ["+str(min(valueset))+", "+str(max(valueset))+" ]\n"
            else :
                info = info + " enum type, distinct values number: " + str(len(valueset))
                info = info + " valueset="+str(valueset)+"\n"
        print(info)





