import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# 输出显示不限制长度
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
from mpl_toolkits.mplot3d import Axes3D
import shap


def main():
    df = pd.read_excel("ElongationFeature.xlsx")
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
    len1 = len(df1.iloc[:, 0])
    # 根据分位数准则删除异常值
    Q1 = df1['Elongation'].quantile(0.25)
    Q3 = df1['Elongation'].quantile(0.75)
    IQR = Q3 - Q1
    # 保留小于极端大的值
    df1 = df1[df1['Elongation'] <= (Q3 + (1.5 * IQR))]
    # 保留大于极端小的值
    df1 = df1[df1['Elongation'] >= (Q1 - (1.5 * IQR))]
    len2 = len(df1.iloc[:, 0])
    remain_index = df1.index.values
    all_index = range(len1)
    del_index = [i for i in all_index if i not in remain_index]
    print('删除的异常值的索引：', del_index)
    print('未进行异常值处理前数据量  ---> ', len1, '\n删除的异常值的个数   ---> ', len1 - len2)
    # 删除了一些行，重置行索引
    df1 = df1.reset_index(drop=True)

    # 删除文本特征，考虑不删除化学式，可以进一步分析误差
    df1.index = df1["formula"].values
    df = df1.drop(["formula", "composition_obj", "Weight Fraction", "Atomic Fraction"], axis=1)
    print(df.head())

    target = 'Elongation'
    # 特征的列名
    features = [i for i in df.columns if i not in [target]]
    Y = df[target]
    X = df[features]

    # 不用这个会造成x和all_data内存地址相同,两个有一个改变就会全变
    all_data = copy.deepcopy(df)
    # 训练集，测试集划分

    Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, test_size=0.2, random_state=59)

    print('Original set  ---> ', X.shape, Y.shape, '\nTraining set  ---> ', Train_X.shape, Train_Y.shape,
          '\nTesting set   ---> ', Test_X.shape, '', Test_Y.shape)

    # Feature Scaling (Standardization)

    std = StandardScaler()

    print('\033[1mStandardardization on Training set'.center(120))
    Train_X_std = std.fit_transform(Train_X)
    Train_X_std = pd.DataFrame(Train_X_std, columns=X.columns, index=Train_X.index)
    # print(Train_X_std.describe())

    print('\n', '\033[1mStandardardization on Testing set'.center(120))
    Test_X_std = std.transform(Test_X)
    Test_X_std = pd.DataFrame(Test_X_std, columns=X.columns, index=Test_X.index)

    best_features = ['MagpieData range MendeleevNumber', 'MagpieData avg_dev MendeleevNumber',
                     'MagpieData avg_dev MeltingT', 'MagpieData mean NValence',
                     'MagpieData mean NsUnfilled', 'MagpieData maximum NUnfilled',
                     'MagpieData maximum GSvolume_pa', 'MagpieData avg_dev GSvolume_pa',
                     'MagpieData minimum SpaceGroupNumber', 'MagpieData mode SpaceGroupNumber',
                     'Lambda entropy', 'Electronegativity local mismatch']
    # 使用最优特征子集建模
    Train_X_std = Train_X_std[best_features]
    Test_X_std = Test_X_std[best_features]

    # 定义一个评价函数
    def Evaluate(model, X, Y):
        model_r2 = CVS(model, X, Y, scoring='r2', cv=10)
        model_mse = CVS(model, X, Y, scoring='neg_mean_squared_error', cv=10)
        model_r2_mean = model_r2.mean()
        model_mse_mean = model_mse.mean()
        model_rmse_mean = np.sqrt(-1 * model_mse_mean)
        print("10折交叉验证r2:", model_r2_mean)
        print("10折交叉验证rmse:", model_rmse_mean)



    params1 = [
        {'C': np.logspace(-3, 3, num=7,base=10), 'gamma': np.logspace(-3, 3,num=7, base=10)}
    ]
    # 根据params1的结果，100,0.1最佳
    params2 = [{'C': np.linspace(50, 150, num=11), 'gamma': np.linspace(0.05, 0.15, num=11)}]
    gridsearch1 = GridSearchCV(SVR(kernel='rbf'), param_grid=params2, scoring='r2', refit=True,
                               return_train_score=True, cv=10)
    gridsearch1.fit(Train_X_std, Train_Y)
    # GridSearchCV的属性
    print('Attrabutes:vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
    # 结果是一个字典，把键打印出来可以按他们查找对应的值
    print('cv_results_:', gridsearch1.cv_results_.keys())
    print('best_estimator_:', gridsearch1.best_estimator_)
    print('best_params_:', gridsearch1.best_params_)
    print('best_score_:', gridsearch1.best_score_)
    print('scorer_:', gridsearch1.scorer_)
    #print('n_splits_:', gridsearch1.n_splits_)





if __name__ == "__main__":
    main()
