import pandas as pd
import numpy as np

def indexA(listA, m):
    """

    :param listA: 列表
    :param m: 要查找索引的元素
    :return: 返回该元素的所有索引
    """
    return [x for (x, y) in enumerate(listA) if y == m]

# 先把data改成corr试试看对不对
def corrDeleteFeature(data, threshold):
    # 计算相关系数矩阵，data是dataframe格式，返回的仍然是dataframe,第一列是目标特征列
    corr = data.corr()
    features = corr.columns
    features_num = len(features)
    pre_deleted_features = []
    for i in range(1, features_num):
        for j in range(i + 1, features_num):
            if corr.iloc[i, j] > threshold:
                # 得到的是成对的特征，一个大列表里很多小列表
                pre_deleted_features.append([features[i], features[j]])
    deleted_features = []
    deleted_features_index = []
    for i in range(len(pre_deleted_features)):
        if i in deleted_features_index:
            continue
        else:
            if corr.loc[pre_deleted_features[i][0], features[0]] >= corr.loc[pre_deleted_features[i][1], features[0]]:
                deleted_features.append(pre_deleted_features[i][1])
                # TODO 获取pre_deleted_features中所有带有这个特征的特征对的索引
                for fe in range(i + 1, len(pre_deleted_features)):
                    if pre_deleted_features[i][1] in pre_deleted_features[fe]:
                        deleted_features_index.append(fe)

            else:
                deleted_features.append(pre_deleted_features[i][0])
                # 删除之后需要把pre_deleted_features中所有带有这个特征的特征对都删除
                for fe in range(i + 1, len(pre_deleted_features)):
                    if pre_deleted_features[i][0] in pre_deleted_features[fe]:
                        deleted_features_index.append(fe)
    return ["deletedFeatures:", deleted_features]

"""
dict={"T":[1,0.3,0.5,0.7,0.8],
    "A":[0.3,1,0.3,0.95,0.92],
    "B":[0.5,0.3,1,0.5,0.91],
    "C":[0.7,0.95,0.5,1,0.8],
    "D": [0.8,0.92,0.91,0.8,1]}
corr=pd.DataFrame(dict,index=["T","A","B","C","D"])
deFe=corrDeleteFeature(corr,0.9)
print(deFe)
"""
"""
a=["A","B","C","D"]
b=["A","B"]
c=[x for x in a if x not in b]
print(c)
"""