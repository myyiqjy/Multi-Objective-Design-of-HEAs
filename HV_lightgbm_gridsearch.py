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
    Y = df[target]
    X = df[features]

    # 不用这个会造成x和all_data内存地址相同,两个有一个改变就会全变
    all_data = copy.deepcopy(df)
    # 训练集，测试集划分
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, test_size=0.2, random_state=1)


    # Feature Scaling (Standardization)

    std = StandardScaler()

    print('\033[1mStandardardization on Training set'.center(120))
    Train_X_std = std.fit_transform(Train_X)
    Train_X_std = pd.DataFrame(Train_X_std, columns=X.columns, index=Train_X.index)
    # print(Train_X_std.describe())

    print('\n', '\033[1mStandardardization on Testing set'.center(120))
    Test_X_std = std.transform(Test_X)
    Test_X_std = pd.DataFrame(Test_X_std, columns=X.columns, index=Test_X.index)

    best_features = ['MagpieData avg_dev CovalentRadius', 'MagpieData mean Electronegativity',
                    'MagpieData avg_dev Electronegativity', 'MagpieData mean NpValence',
                    'MagpieData mean NUnfilled', 'MagpieData avg_dev SpaceGroupNumber',
                    'Mean cohesive energy', 'Shear modulus strength model']

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



    #调参，先提高准确率，调整max_depth，num_leaves，learning_rate
    # 先把学习率定为一个较高的值0.1（收敛速度会比较快），调整max_depth 和 num_leaves
    lgbm = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
                             metric='rmse', random_state=100)

    params_test1={
        # 提高精确度
        'n_estimators':range(500,1200,100),
        'max_depth': range(6,15),
        'num_leaves':range(10, 100, 10),

    }
    params_test2 = {
        # 提高精确度
        'n_estimators': range(650, 750, 10),
        'max_depth': [12],
        'num_leaves': range(15, 25),
        'learning_rate':[0.1],
    }
    #verbosity：default=1, type=int, alias=verbose
    #表示是否输出中间信息，小于0 ，仅仅输出致命的, 等于0 ，还会输出错误 (警告)信息, 大于0 ，则还会输出info信息.
    gridsearch = GridSearchCV(estimator=lgbm, param_grid=params_test2, scoring='neg_mean_squared_error', cv=10, verbose=-1, n_jobs=4)
    gridsearch.fit(Train_X_std, Train_Y)
    print('cv_results_:', gridsearch.cv_results_.keys())
    print('best_estimator_:', gridsearch.best_estimator_)
    print('best_params_:', gridsearch.best_params_)
    print('best_score_:', gridsearch.best_score_)
    print('scorer_:', gridsearch.scorer_)


    """
    #调参，降低过拟合
    lgbm_tiaocan2 = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
                              learning_rate=0.1,max_depth=12,n_estimators=690,num_leaves=15,
                            metric='rmse',random_state=100) 
     # bagging_fraction + bagging_freq参数必须同时设置,bagging_freq默认0
    params_test3={
    # 下面3个参数都是降低过拟合的
        'min_data_in_leaf':range(18,23),
        'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'bagging_fraction': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'bagging_freq': [int(i) for i in np.linspace(0, 50, num=25)]
    }
    
    
    
    gridsearch2 = GridSearchCV(estimator=lgbm_tiaocan2, param_grid=params_test3, scoring='neg_mean_squared_error', cv=10, verbose=-1, n_jobs=4)
    gridsearch2.fit(Train_X_std, Train_Y)
    print('cv_results_:', gridsearch2.cv_results_.keys())
    print('best_estimator_:', gridsearch2.best_estimator_)
    print('best_params_:', gridsearch2.best_params_)
    print('best_score_:', gridsearch2.best_score_)
    print('scorer_:', gridsearch2.scorer_)
    """

    """
    # 调节学习率
    lgbm_tiaocan1 = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
                              max_depth=12,n_estimators=690,num_leaves=15,
                            feature_fraction=1,min_data_in_leaf=20,metric='rmse',random_state=100)
    params_test1={
        'learning_rate':np.linspace(0.01, 0.3, num=30),
    }
    gridsearch3 = GridSearchCV(estimator=lgbm_tiaocan1, param_grid=params_test1, scoring='neg_mean_squared_error', cv=10, verbose=-1, n_jobs=4)
    gridsearch3.fit(Train_X_std, Train_Y)
    print('best_estimator_:', gridsearch3.best_estimator_)
    print('best_params_:', gridsearch3.best_params_)
    print('best_score_:', gridsearch3.best_score_)
    print('scorer_:', gridsearch3.scorer_)
    """


    # 最优模型的交叉验证结果
    # 最终的模型
    lgbm_last = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
                             max_depth=12, num_leaves=15, learning_rate=0.1,
                             n_estimators=690, feature_fraction=1,min_data_in_leaf=20,
                             metric='rmse',random_state=100)

    Evaluate(lgbm_last,Train_X_std,Train_Y)




if __name__ == "__main__":
    main()
