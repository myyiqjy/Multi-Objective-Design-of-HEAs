import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score as CVS
import numpy as np
import shap
import time
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
### 模型选择
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import copy
import lightgbm as lgb

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVR
import random

# 输出显示不限制长度
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
import geatpy as ea


def main():
    def Evaluate(model, X, Y):
        model_r2 = CVS(model, X, Y, scoring='r2', cv=10)
        model_mse = CVS(model, X, Y, scoring='neg_mean_squared_error', cv=10)
        model_r2_mean = model_r2.mean()
        model_mse_mean = model_mse.mean()
        model_rmse_mean = np.sqrt(-1 * model_mse_mean)
        print("10折交叉验证r2:", model_r2_mean)
        print("10折交叉验证rmse:", model_rmse_mean)

    def Evaluate_one(model, Train_X, Train_Y, Test_X, Test_Y):
        model.fit(Train_X, Train_Y)
        pred1 = model.predict(Train_X)
        pred2 = model.predict(Test_X)
        print("Train_r2", round(r2_score(Train_Y, pred1), 2))
        print("Train_rmse", round(np.sqrt(mean_squared_error(Train_Y, pred1)), 2))
        print("Test_r2", round(r2_score(Test_Y, pred2), 2))
        print("Test_rmse", round(np.sqrt(mean_squared_error(Test_Y, pred2)), 2))

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
    # print(HV_Train_X_std.head())

    # 建立两个目标函数
    # 硬度模型
    lgbm_HV = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
                                max_depth=12, num_leaves=15, learning_rate=0.1,
                                n_estimators=690, feature_fraction=1, min_data_in_leaf=20,
                                metric='rmse', random_state=100)
    # Evaluate(lgbm_HV, HV_Train_X_std, Train_Y_HV)
    # Evaluate_one(lgbm_HV, HV_Train_X_std, Train_Y_HV, HV_Test_X_std, Test_Y_HV)
    lgbm_HV.fit(HV_Train_X_std, Train_Y_HV)
    # 压缩应变模型
    svr_EL = SVR(kernel='rbf', C=110, gamma=0.07)
    # Evaluate(svr_EL, EL_Train_X_std, Train_Y_EL)
    # Evaluate_one(svr_EL, EL_Train_X_std, Train_Y_EL, EL_Test_X_std, Test_Y_EL)
    svr_EL.fit(EL_Train_X_std, Train_Y_EL)

    # 需要把数据转换成一行，reshape(-1,1)代表一列
    # print(svr_EL.predict(np.array(min_feature).reshape(1,-1)))

    class MyProblem(ea.Problem):  # 继承Problem父类

        def __init__(self, M=2):
            name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
            Dim = 20  # 初始化Dim（决策变量维数）
            maxormins = [-1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
            varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
            lb = min_feature  # 决策变量下界
            ub = max_feature  # 决策变量上界
            lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
            ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
            # 调用父类构造方法完成实例化
            ea.Problem.__init__(self,
                                name,
                                M,
                                maxormins,
                                Dim,
                                varTypes,
                                lb,
                                ub,
                                lbin,
                                ubin)

        def evalVars(self, Vars):  # 目标函数
            x1 = Vars[:, [0]]
            x2 = Vars[:, [1]]
            x3 = Vars[:, [2]]
            x4 = Vars[:, [3]]
            x5 = Vars[:, [4]]
            x6 = Vars[:, [5]]
            x7 = Vars[:, [6]]
            x8 = Vars[:, [7]]
            x9 = Vars[:, [8]]
            x10 = Vars[:, [9]]
            x11 = Vars[:, [10]]
            x12 = Vars[:, [11]]
            x13 = Vars[:, [12]]
            x14 = Vars[:, [13]]
            x15 = Vars[:, [14]]
            x16 = Vars[:, [15]]
            x17 = Vars[:, [16]]
            x18 = Vars[:, [17]]
            x19 = Vars[:, [18]]
            x20 = Vars[:, [19]]

            HV = np.array(lgbm_HV.predict(Vars))
            EL = np.array(svr_EL.predict(Vars))


            # np.hstack将参数元组的元素数组按水平方向进行叠加
            f = np.hstack([HV, EL])
            #print(f)
            f = f.reshape(-1, 2, order='F')
            #print(f)
            return f

        def calReferObjV(self):  # 计算全局最优解，如果实际问题并不知道最优解，可省略，这是一个参考值
            # 欲得到500个参考值
            N = 100
            Var_refer_sort=np.ones((10,20))
            Var_refer_rand=np.ones((N,20))
            for i in range(20):
                # 每个特征都根据最小值，最大值等间隔划分10份
                Var_refer_sort[:,i]=np.linspace(min_feature[i],max_feature[i],num=10)
            for i in range(N):
                # 对每一行随机选择改变10列的值
                change_index = np.random.randint(0, 19, size=10)
                for j in change_index:
                    Var_refer_rand[i,j]=Var_refer_sort[random.randint(0,9),j]
            HV = np.array(lgbm_HV.predict(Var_refer_rand))
            EL = np.array(svr_EL.predict(Var_refer_rand))
            f_refer = np.hstack([HV, EL])
            f_refer = f_refer.reshape(-1, 2, order='F')
            #print("refer",f_refer[:50,:])
            return f_refer

    # 实例化问题对象
    problem = MyProblem()
    # 构建算法
    algorithm = ea.moea_NSGA2_templet(
        problem,
        # 种群的大小为NIND
        ea.Population(Encoding='BG', NIND=200),
        MAXGEN=15000,  # 最大进化代数
        logTras=1)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    algorithm.mutOper.Pm = 0.1  # 修改变异算子的变异概率
    algorithm.recOper.XOVR = 0.5  # 修改交叉算子的交叉概率
    # 求解
    # drawing取1只显示最终结果，2实时绘制目标空间动态图，3实时绘制决策空间动态图
    # 设置为2时图片太多了
    # drawlog是否根据日志绘制迭代变化图像
    res = ea.optimize(algorithm,
                      verbose=False,
                      drawing=1,
                      outputMsg=True,
                      drawLog=True,
                      saveFlag=True)
    print(res)



if __name__ == "__main__":
    main()
