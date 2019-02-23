from data import *
from model import *

#主函数
#author: luo
#date: 2018-09-14,Fri,Sun
#原理:https://github.com/liudragonfly/GBDT/blob/master/GBDT.ipynb
"""
前20次训练信息
iter:    1 : train loss=0.385024
iter:    2 : train loss=0.250209
iter:    3 : train loss=0.171565
iter:    4 : train loss=0.124508
iter:    5 : train loss=0.095248
iter:    6 : train loss=0.075281
iter:    7 : train loss=0.055675
iter:    8 : train loss=0.043991
iter:    9 : train loss=0.036140
iter:   10 : train loss=0.031209
iter:   11 : train loss=0.024821
iter:   12 : train loss=0.021210
iter:   13 : train loss=0.017955
iter:   14 : train loss=0.016377
iter:   15 : train loss=0.013434
iter:   16 : train loss=0.010784
iter:   17 : train loss=0.009160
iter:   18 : train loss=0.008423
iter:   19 : train loss=0.006448
iter:   20 : train loss=0.005265
"""

if __name__ == '__main__' :
    data_file = "credit.data.csv"
    dataset = dataset(data_file)
    gbdt = GBDT(max_iter=20, sample_rate=0.8, learn_rate=0.5, max_depth=7, loss_type='binary-classification')
    gbdt.fit(dataset, dataset.get_instances_idset())