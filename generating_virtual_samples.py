import pandas as pd
import numpy as np
from random import randint
import random
import matplotlib.pyplot as plt

# 生成虚拟样本
# HV_elements=["C", "Al", "Si", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Y", "Zr", "Nb", "Mo", "Sn", "Nd", "Hf", "Ta", "W", "Re"]
# HV_count=[4, 336, 7, 2, 107, 58, 369, 45, 404, 328, 382, 209, 2, 34, 49, 78, 2, 1, 28, 27, 13, 5]
# EL_elements=["C", "Al", "Si", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Y", "Zr", "Nb", "Mo", "Pd", "Sn", "Hf", "Ta", "W", "Re"]
# EL_count=[4, 85, 6, 2, 85, 44, 107, 26, 122, 91, 115, 46, 2, 30, 52, 47, 4, 4, 25, 20, 14, 5]
all_elements = ["C", "Al", "Si", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Y", "Zr", "Nb", "Mo", "Sn", "Pd",
                "Nd", "Hf", "Ta", "W", "Re"]
HV_count = [4, 336, 7, 2, 107, 58, 369, 45, 404, 328, 382, 209, 2, 34, 49, 78, 2, 0, 1, 28, 27, 13, 5]
EL_count = [4, 85, 6, 2, 85, 44, 107, 26, 122, 91, 115, 46, 2, 30, 52, 47, 4, 4, 0, 25, 20, 14, 5]
# 计算各个元素出现的概率
all_elements_prob = [0.5 * x / sum(HV_count) + 0.5 * y / sum(EL_count) for x, y in zip(HV_count, EL_count)]
print([str(round(all_elements_prob[i]*100,1))+'%' for i in range(len(all_elements))])
#print(sum(all_elements_prob))

# 绘制饼图，准备数据
labels = all_elements
sizes = all_elements_prob
# 设置随机种子，以确保每次生成相同的颜色
np.random.seed(0)
# 生成23种暖色、清新颜色
colors = []
num_colors = len(all_elements)

for _ in range(num_colors):
    # 生成RGB颜色值
    r = np.random.uniform(0.7, 1)  # 红色通道范围：0.7-1
    g = np.random.uniform(0.8, 1)  # 绿色通道范围：0.8-1
    b = np.random.uniform(0.8, 1)  # 蓝色通道范围：0.9-1

    colors.append((r, g, b))
# 绘制饼图
for i in range(len(labels)):
    if sizes[i] > 0.05:
        labels[i] = all_elements[i]
    else:
        labels[i] = ''
plt.pie(sizes, labels=labels, colors=colors, autopct=lambda x: '%1.1f%%' % x if x >= 5 else '', startangle=90,pctdistance=0.95)
#plt.legend()
plt.axis('equal')

# 设置图表标题
#plt.title('Fruit distribution')

# 显示图表
plt.savefig('original_el_pie.png', dpi=500)
plt.show()


# 轮盘赌选择元素
def roulette(probability, num):
    """

    :param probability: 对应的每个元素出现的概率
    :param num: 运行次数，选择多少个元素
    :return:
    """
    result = []
    probabilityTotal = np.zeros(len(probability))
    probabilityTmp = 0
    # 计算每一个元素的截止概率值，每个元素都对应一个概率区间
    for i in range(len(probability)):
        probabilityTmp += probability[i]
        probabilityTotal[i] = probabilityTmp
    while len(result) < num:
        randomNumber = np.random.rand()
        for i in range(1, len(probabilityTotal)):
            if randomNumber < probabilityTotal[0]:
                result.append(0)
                # print("random number:", randomNumber, "<index 0:", probabilityTotal[0])
                break
            elif probabilityTotal[i - 1] < randomNumber <= probabilityTotal[i]:
                result.append(i)
                # print("index ", i - 1, ":", probabilityTotal[i - 1], "<random number:", randomNumber, "<index ", i, ":",
                # probabilityTotal[i])
        result = list(set(result))
    return result

HV_fenweishu = [[0.051, 0.0825, 0.10200000000000001, 0.11525, 0.128], [0.017, 0.12, 0.182, 0.24625, 0.47],
                [0.024, 0.0645, 0.111, 0.158, 0.184],
                [0.2, 0.21250000000000002, 0.225, 0.2375, 0.25], [0.033, 0.1405, 0.2, 0.2365, 0.444],
                [0.016, 0.15725, 0.196, 0.22075, 0.332],
                [0.025, 0.161, 0.186, 0.217, 0.35], [0.057, 0.172, 0.192, 0.222, 0.4],
                [0.048, 0.154, 0.182, 0.213, 0.571], [0.048, 0.16075, 0.1845, 0.222, 0.429],
                [0.05, 0.154, 0.191, 0.238, 0.5], [0.02, 0.096, 0.158, 0.2, 0.332],
                [0.2, 0.21250000000000002, 0.225, 0.2375, 0.25], [0.125, 0.18375, 0.2075, 0.25, 0.375],
                [0.02, 0.182, 0.204, 0.235, 0.375], [0.01, 0.098, 0.14850000000000002, 0.2, 0.286],
                [0.017, 0.02175, 0.0265, 0.03125, 0.036], [0, 0, 0, 0, 0],
                [0.167, 0.167, 0.167, 0.167, 0.167], [0.091, 0.111, 0.17049999999999998, 0.2, 0.333],
                [0.051, 0.17049999999999998, 0.2, 0.2425, 0.333],
                [0.048, 0.2, 0.222, 0.235, 0.286], [0.106, 0.111, 0.116, 0.128, 0.143]]

EL_fenweishu = [[0.051, 0.0825, 0.10200000000000001, 0.11525, 0.128], [0.017, 0.14, 0.167, 0.2, 0.333],
                [0.024, 0.064, 0.0905, 0.13949999999999999, 0.184],
                [0.2, 0.21250000000000002, 0.225, 0.2375, 0.25], [0.033, 0.143, 0.182, 0.211, 0.333],
                [0.008, 0.111, 0.182, 0.21375, 0.429], [0.025, 0.167, 0.182, 0.2, 0.312],
                [0.038, 0.176, 0.2, 0.216, 0.328], [0.1, 0.167, 0.191, 0.22075, 0.347],
                [0.048, 0.172, 0.192, 0.2245, 0.333], [0.111, 0.172, 0.2, 0.243, 0.333],
                [0.02, 0.143, 0.167, 0.1945, 0.328], [0.2, 0.21250000000000002, 0.225, 0.2375, 0.25],
                [0.143, 0.182, 0.212, 0.25, 0.333], [0.029, 0.18, 0.202, 0.23725, 0.333],
                [0.01, 0.155, 0.2, 0.222, 0.286], [0.017, 0.03125, 0.0465, 0.0655, 0.091],
                [0.172, 0.17725, 0.182, 0.18675, 0.192], [0, 0, 0, 0, 0], [0.024, 0.111, 0.167, 0.2, 0.333],
                [0.051, 0.161, 0.191, 0.2, 0.25], [0.048, 0.17525000000000002, 0.2175, 0.24575, 0.286],
                [0.106, 0.111, 0.116, 0.128, 0.143]]

elements_range = [[] for i in range(len(all_elements))]
for i in range(len(all_elements)):
    if HV_fenweishu[i][4] == 0:
        elements_range[i].append(EL_fenweishu[i][0])
        elements_range[i].append(EL_fenweishu[i][4])
    if EL_fenweishu[i][4] == 0:
        elements_range[i].append(HV_fenweishu[i][0])
        elements_range[i].append(HV_fenweishu[i][4])
    else:
        elements_range[i].append(max(HV_fenweishu[i][0], EL_fenweishu[i][0]))
        elements_range[i].append(min(HV_fenweishu[i][4], EL_fenweishu[i][4]))
#print(elements_range)

def elements_fraction(elements_range,index):
    """

    :param elements_range: 传入每个元素的范围
    :param index: 传入选择的元素索引，可以是一个列表
    :return:
    """
    result=np.ones(len(index))
    for i in range(len(index)):
        index_temp=index[i]
        result[i]=elements_range[index_temp][0]+(elements_range[index_temp][1]-elements_range[index_temp][0])*random.random()
    result=[round(i/sum(result),2) for i in result]
    return result


def generate_virture_samples(elements_list,elements_range,elements_prob,num):
    """

    :param elements_list: 待选择的元素集合
    :param elements_range: 元素范围
    :param num: 需要生成虚拟样本的个数
    :return:
    """

    formula_list=[str(i) for i in range(num)]
    for i in range(num):
        yuan_num = randint(4, 7)
        #print(yuan_num)
        index=roulette(elements_prob, yuan_num)
        #print(index)
        elements = []
        for j in range(yuan_num):
            elements.append(elements_list[index[j]])
        #print(elements)
        elements_frac=elements_fraction(elements_range,index)
        elements_frac = [str(i) for i in elements_frac]
        formula=[]
        for k in range(yuan_num):
            formula.append(elements[k])
            formula.append(elements_frac[k])
        #print(formula)
        formula = ''.join(formula)
        #print(formula)
        formula_list[i]=formula

    formula_df=pd.DataFrame(formula_list,columns=["formula"])

    return formula_df

df=generate_virture_samples(all_elements,elements_range,elements_prob=all_elements_prob,num=100000)
#print(df)
df.to_excel("Virture_samples_100000.xlsx")


