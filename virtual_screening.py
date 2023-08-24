import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score as CVS
import numpy as np
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
### 模型选择
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import copy
import lightgbm as lgb

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import random
from compute_distance import *

# 输出显示不限制长度
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)


def main():
    df = pd.read_excel("HV_Feature.xlsx")
    # 如果有重复值，则保留第一个
    len1 = len(df.iloc[:, 0])
    df.drop_duplicates(keep='first', inplace=True)
    len2 = len(df.iloc[:, 0])
    remain_index = df.index.values
    all_index = range(len1)
    del_index = [i for i in all_index if i not in remain_index]
    print('删除的重复值的索引：', del_index)
    print('Original set  ---> ', len1, '\ndroped duplicates   ---> ', len1 - len2)

    df1 = df.copy()

    # 删除这两个文本特征
    df1.index = df1["Composition"].values
    df = df1.drop(["Composition", "composition_obj", "Weight Fraction", "Atomic Fraction"], axis=1)

    target = 'HV'
    # 特征的列名
    features = [i for i in df.columns if i not in [target]]
    Y_HV = df[target]
    X_HV = df[features]

    # 不用这个会造成x和all_data内存地址相同,两个有一个改变就会全变
    all_data = copy.deepcopy(df)
    # 训练集，测试集划分
    Train_X_HV, Test_X_HV, Train_Y_HV, Test_Y_HV = train_test_split(X_HV, Y_HV, test_size=0.2, random_state=1)
    best_features_HV = ['MagpieData avg_dev CovalentRadius', 'MagpieData mean Electronegativity',
                        'MagpieData avg_dev Electronegativity', 'MagpieData mean NpValence',
                        'MagpieData mean NUnfilled', 'MagpieData avg_dev SpaceGroupNumber',
                        'Mean cohesive energy', 'Shear modulus strength model']
    best_features_elongation = ['MagpieData range MendeleevNumber', 'MagpieData avg_dev MendeleevNumber',
                                'MagpieData avg_dev MeltingT', 'MagpieData mean NValence',
                                'MagpieData mean NsUnfilled', 'MagpieData maximum NUnfilled',
                                'MagpieData maximum GSvolume_pa', 'MagpieData avg_dev GSvolume_pa',
                                'MagpieData minimum SpaceGroupNumber', 'MagpieData mode SpaceGroupNumber',
                                'Lambda entropy', 'Electronegativity local mismatch']
    HV_Train_X_feature_HV = Train_X_HV[best_features_HV]
    HV_Train_X_feature_Elongation = Train_X_HV[best_features_elongation]
    HV_Test_X_feature_HV = Test_X_HV[best_features_HV]
    HV_Test_X_feature_Elongation = Test_X_HV[best_features_elongation]

    df = pd.read_excel("ElongationFeature.xlsx")
    # 如果有重复值，则保留第一个
    df.drop_duplicates(keep='first', inplace=True)
    df1 = df.copy()
    # 根据分位数准则删除异常值
    Q1 = df1['Elongation'].quantile(0.25)
    Q3 = df1['Elongation'].quantile(0.75)
    IQR = Q3 - Q1
    # 保留小于极端大的值
    df1 = df1[df1['Elongation'] <= (Q3 + (1.5 * IQR))]
    # 保留大于极端小的值
    df1 = df1[df1['Elongation'] >= (Q1 - (1.5 * IQR))]

    # 删除了一些行，重置行索引
    df1 = df1.reset_index(drop=True)

    # 删除文本特征，考虑不删除化学式，可以进一步分析误差
    df1.index = df1["formula"].values
    df = df1.drop(["formula", "composition_obj", "Weight Fraction", "Atomic Fraction"], axis=1)
    print(df.head())

    target = 'Elongation'
    # 特征的列名
    features = [i for i in df.columns if i not in [target]]
    Y_EL = df[target]
    X_EL = df[features]

    # 不用这个会造成x和all_data内存地址相同,两个有一个改变就会全变
    all_data = copy.deepcopy(df)
    # 训练集，测试集划分

    Train_X_EL, Test_X_EL, Train_Y_EL, Test_Y_EL = train_test_split(X_EL, Y_EL, test_size=0.2, random_state=59)
    EL_Train_X_feature_HV = Train_X_EL[best_features_HV]
    EL_Train_X_feature_EL = Train_X_EL[best_features_elongation]
    EL_Test_X_feature_HV = Test_X_EL[best_features_HV]
    EL_Test_X_feature_EL = Test_X_EL[best_features_elongation]

    # Feature Scaling (Standardization)
    std_HV = StandardScaler()
    std_EL = StandardScaler()

    print('\033[1mStandardardization on Training set'.center(120))
    HV_Train_X_std_feature_HV = std_HV.fit_transform(HV_Train_X_feature_HV)
    HV_Train_X_std_feature_HV = pd.DataFrame(HV_Train_X_std_feature_HV, columns=best_features_HV,
                                             index=Train_X_HV.index)
    print("=====================HV_Train_X_std_feature_HV.describe==========================")
    HV_Train_X_std_feature_HV_describe = HV_Train_X_std_feature_HV.describe()
    # HV_Train_X_std_feature_HV_describe.to_excel("HV_Train_X_std_feature_HV_describe.xlsx", index=True)

    EL_Train_X_std_feature_EL = std_EL.fit_transform(EL_Train_X_feature_EL)
    EL_Train_X_std_feature_EL = pd.DataFrame(EL_Train_X_std_feature_EL, columns=best_features_elongation,
                                             index=Train_X_EL.index)
    print("=====================EL_Train_X_std_feature_EL.describe==========================")
    EL_Train_X_std_feature_EL_describe = EL_Train_X_std_feature_EL.describe()
    # EL_Train_X_std_feature_EL_describe.to_excel("EL_Train_X_std_feature_EL_describe.xlsx",index=True)

    min_feature = []
    max_feature = []
    for i in range(len(best_features_HV)):
        min_feature.append(min(HV_Train_X_std_feature_HV.iloc[:, i]))
        max_feature.append(max(HV_Train_X_std_feature_HV.iloc[:, i]))
    for i in range(len(best_features_elongation)):
        min_feature.append(min(EL_Train_X_std_feature_EL.iloc[:, i]))
        max_feature.append(max(EL_Train_X_std_feature_EL.iloc[:, i]))
    # print(min_feature)
    # print(max_feature)

    print('\n', '\033[1mStandardardization on Testing set'.center(120))
    HV_Test_X_std_feature_HV = std_HV.transform(HV_Test_X_feature_HV)
    HV_Test_X_std_feature_HV = pd.DataFrame(HV_Test_X_std_feature_HV, columns=best_features_HV, index=Test_X_HV.index)
    HV_Test_X_std_feature_EL = std_EL.transform(HV_Test_X_feature_Elongation)
    HV_Test_X_std_feature_EL = pd.DataFrame(HV_Test_X_std_feature_EL, columns=best_features_elongation,
                                            index=Test_X_HV.index)
    HV_Train_X_std_feature_EL = std_EL.transform(HV_Train_X_feature_Elongation)
    HV_Train_X_std_feature_EL = pd.DataFrame(HV_Train_X_std_feature_EL, columns=best_features_elongation,
                                             index=Train_X_HV.index)
    EL_Test_X_std_feature_HV = std_HV.transform(EL_Test_X_feature_HV)
    EL_Test_X_std_feature_HV = pd.DataFrame(EL_Test_X_std_feature_HV, columns=best_features_HV, index=Test_X_EL.index)
    EL_Train_X_std_feature_HV = std_HV.transform(EL_Train_X_feature_HV)
    EL_Train_X_std_feature_HV = pd.DataFrame(EL_Train_X_std_feature_HV, columns=best_features_HV,
                                             index=Train_X_EL.index)
    EL_Test_X_std_feature_EL = std_EL.transform(EL_Test_X_feature_EL)
    EL_Test_X_std_feature_EL = pd.DataFrame(EL_Test_X_std_feature_EL, columns=best_features_elongation,
                                            index=Test_X_EL.index)

    # 合并数据集
    HV_Train_X_std = pd.concat([HV_Train_X_std_feature_HV, HV_Train_X_std_feature_EL], axis=1)
    HV_Test_X_std = pd.concat([HV_Test_X_std_feature_HV, HV_Test_X_std_feature_EL], axis=1)
    EL_Train_X_std = pd.concat([EL_Train_X_std_feature_HV, EL_Train_X_std_feature_EL], axis=1)
    EL_Test_X_std = pd.concat([EL_Test_X_std_feature_HV, EL_Test_X_std_feature_EL], axis=1)



    # ========================虚拟样本筛选==========================================
    df_virtual = pd.read_excel("Virture_samples_Feature_100000.xlsx")
    df_virtual["formula"]=pd.read_excel("Virture_samples_100000_formula_sorted.xlsx")
    # 如果有重复值，则保留第一个
    len1 = len(df_virtual.iloc[:, 0])
    df_virtual.drop_duplicates(keep='first', inplace=True)
    len2 = len(df_virtual.iloc[:, 0])
    df2 = df_virtual.copy()
    print('Original set  ---> ', len1, '\ndroped duplicates   ---> ', len1 - len2)
    # 删除了一些行，重置行索引
    df2 = df2.reset_index(drop=True)

    # 删除文本特征，考虑不删除化学式，可以进一步分析误差
    df2.index = df2["formula"].values
    df2 = df2.drop(["formula"], axis=1)
    Virtual_X_feature_HV = df2[best_features_HV]
    Virtual_X_feature_EL = df2[best_features_elongation]
    Virtual_X_std_feature_HV = std_HV.transform(Virtual_X_feature_HV)
    Virtual_X_std_feature_HV = pd.DataFrame(Virtual_X_std_feature_HV, columns=best_features_HV,
                                            index=df2.index)
    Virtual_X_std_feature_EL = std_EL.transform(Virtual_X_feature_EL)
    Virtual_X_std_feature_EL = pd.DataFrame(Virtual_X_std_feature_EL, columns=best_features_elongation,
                                            index=df2.index)
    Virtual_X_std = pd.concat([Virtual_X_std_feature_HV, Virtual_X_std_feature_EL], axis=1)
    num_samples=5
    filtered_samples_index, filtered_samples_distance = compute_distance(Virtual_X_std, pareto_solvers=Pareto_solves,
                                                                         num_samples=num_samples)
    filtered_samples_index = [j for i in filtered_samples_index for j in i]
    # print(filtered_samples_distance)
    #print(filtered_samples_index)
    df_solvers = pd.DataFrame(filtered_samples_index,columns=["formula"])
    # 按列展开
    df_solvers["distance"]=filtered_samples_distance.reshape((-1,1),order='F')
    df_solvers["HV"] = lgbm_HV.predict(Virtual_X_std.loc[filtered_samples_index])
    df_solvers["EL"] = svr_EL.predict(Virtual_X_std.loc[filtered_samples_index])
    df_solvers["obj_hv"]=[i for i in Obj[:,0] for j in range(num_samples)]
    df_solvers["obj_el"] = [i for i in Obj[:, 1] for j in range(num_samples)]
    df_solvers.to_excel("filterd_virtual_samples_predict_objv_100000.xlsx", index=True)


if __name__ == "__main__":
    main()
