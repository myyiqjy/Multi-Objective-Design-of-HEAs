import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.feature_selection import VarianceThreshold
from CorrDeleteFeature import corrDeleteFeature
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_predict
from genetic_selection import GeneticSelectionCV
from matplotlib.pyplot import MultipleLocator
import lightgbm as lgb

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
    print('Original set  ---> ', len1, '\ndroped duplicates   ---> ', len1 - len2)
    # 删除这两个文本特征
    df = df.drop(["Composition", "composition_obj", "Weight Fraction", "Atomic Fraction"], axis=1)



    target = 'HV'
    # 特征的列名
    features = [i for i in df.columns if i not in [target]]
    Y = df[target]
    X = df[features]

    # 不用这个会造成x和all_data内存地址相同,两个有一个改变就会全变
    all_data = copy.deepcopy(df)
    # 训练集，测试集划分
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, test_size=0.2, random_state=1)
    Train_X.reset_index(drop=True, inplace=True)
    Test_X.reset_index(drop=True, inplace=True)
    Train_Y.reset_index(drop=True, inplace=True)
    Test_Y.reset_index(drop=True, inplace=True)

    print('Original set  ---> ', X.shape, Y.shape, '\nTraining set  ---> ', Train_X.shape, Train_Y.shape,
          '\nTesting set   ---> ', Test_X.shape, '', Test_Y.shape)

    # Feature Scaling (Standardization)

    std = StandardScaler()

    print('\033[1mStandardardization on Training set'.center(120))
    Train_X_std = std.fit_transform(Train_X)
    Train_X_std = pd.DataFrame(Train_X_std, columns=X.columns)
    # print(Train_X_std.describe())

    print('\n', '\033[1mStandardardization on Testing set'.center(120))
    Test_X_std = std.transform(Test_X)
    Test_X_std = pd.DataFrame(Test_X_std, columns=X.columns)

    # print(Test_X_std.describe())

    def variance_threshold_selector(data, threshold=0):
        selector = VarianceThreshold(threshold)
        selector.fit(data)
        return data[data.columns[selector.get_support(indices=True)]]

    FeatureNum = len(Train_X_std.columns)
    Train_X_std = variance_threshold_selector(Train_X_std)
    VFeatureNum = len(Train_X_std.columns)
    features = Train_X_std.columns
    Test_X_std = Test_X_std[features]
    # VarianceThreshold deleted feature num  --->  17
    print('Original feature num  ---> ', FeatureNum, '\nVarianceThreshold deleted feature num  ---> ',
          FeatureNum - VFeatureNum)

    print('\n', '\033[1m相关系数特征选择'.center(120))
    # 先把特征和目标变量拼成一个dataframe
    data_Train_X_Y = pd.concat([Train_Y, Train_X_std], axis=1)
    # 使用我在CorrDeleteFeature.py编写的函数实现相关系数筛选特征，['deletedFeatures:', ['A', 'B']]，返回的特征名索引是1
    deletedFeatureName = corrDeleteFeature(data_Train_X_Y, 0.9)
    # Original feature num  --->  144
    # After PearsonCorrelationThreshold deleting feature num  --->  80
    # 皮尔逊相关系数法删除了64个特征
    print('Original feature num  ---> ', VFeatureNum, '\nAfter PearsonCorrelationThreshold deleting feature num  ---> ',
          VFeatureNum - len(deletedFeatureName[1]))
    features = [x for x in features if x not in deletedFeatureName[1]]
    Train_X_std = Train_X_std[features]
    Test_X_std = Test_X_std[features]

    # 模型构建
    print('\n', '\033[1mLightGBM模型构建'.center(120))
    from sklearn.model_selection import cross_val_score as CVS
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from time import time


    lgbm = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
                         metric='rmse', random_state=100)
    select_feature_all_times=[]
    lgbm_avg_mse=[]
    tic_fwd = time()
    sfs_forward = SFS(lgbm, k_features=len(features), forward=True,
                    cv=10,scoring='neg_mean_squared_error')
    sfs_forward.fit(Train_X_std, Train_Y)
    sfs_results=sfs_forward.subsets_
    for i in range(len(features)):
        select_feature_all_times.append(sfs_results[i+1]['feature_names'])
        lgbm_avg_mse.append(sfs_results[i+1]['avg_score'])
    toc_fwd = time()
    lgbm_avg_mse=[-1*i for i in svr_avg_mse]
    lgbm_avg_rmse=[np.sqrt(i) for i in svr_avg_mse]
    print(f"Done in {toc_fwd - tic_fwd:.3f}s")
    print('SFS RMSE  ---> ',svr_avg_rmse,'\n特征名  ---> ',select_feature_all_times)

    min_indx=np.argmin(lgbm_avg_rmse)#min value index
    # 打印最佳特征子集
    best_features=select_feature_all_times[min_indx]
    print('SFS svr 最佳特征子集-------> ',best_features,'\n最佳特征子集数量-------> ',min_indx+1)




    # 包装法特征筛选
    print('\n', '\033[1m遗传算法结合SVR模型进行特征筛选'.center(120))
    svr = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
                         metric='rmse', random_state=100)

    # 因为遗传算法的随机性，重复运行20次
    # 第一次max10,20
    # 第二次max9,5
    # 第三次max8,5
    # 第四次max7,5
    # 第五次max6,5
    # 第六次max5,5
    # 第六次max4,5
    featureNum = []
    genetic_RMSE = []
    select_feature_all_times = []
    for i in range(5):
        selector = GeneticSelectionCV(svr,
                                      cv=10,
                                      verbose=0,
                                      scoring="neg_mean_squared_error",
                                      max_features=4,
                                      n_population=200,
                                      crossover_proba=0.5,
                                      mutation_proba=0.2,
                                      n_generations=200,
                                      crossover_independent_proba=0.5,
                                      mutation_independent_proba=0.05,
                                      tournament_size=3,
                                      n_gen_no_change=10,
                                      caching=True,
                                      n_jobs=-1)
        selector = selector.fit(Train_X_std, Train_Y)
        print("=============================第%s次结果==============================" % (i + 1))
        print('有效变量的数量：', selector.n_features_)
        featureNum.append(selector.n_features_)
        # print('np.array(selector.population_).shape------->',np.array(selector.population_).shape)
        print('selector.generation_scores_--------->', np.sqrt(-1 * selector.generation_scores_[-1]))
        genetic_RMSE.append(np.sqrt(-1 * selector.generation_scores_[-1]))
        best_features = Train_X_std.columns.values[selector.support_]
        # print('最佳特征子集--------->',best_features)
        select_feature_all_times.append(best_features)

    print('genetic svr 特征数-------> ', featureNum)
    print('genetic svr RMSE-------> ', genetic_RMSE)
    min_indx = np.argmin(genetic_RMSE)  # min value index
    # 最少的特征数量可能有多个，这里只打印了第一个
    min_feature_indx = np.argmin(featureNum)
    # 打印最佳特征子集
    best_features = select_feature_all_times[min_indx]
    min_features = select_feature_all_times[min_feature_indx]
    # print('genetic svr 最佳特征子集-------> ', best_features, '\n最佳特征子集数量-------> ', featureNum[min_indx])
    print('genetic svr 最佳特征子集-------> ', min_features, '\n最佳特征子集数量-------> ', featureNum[min_feature_indx])


if __name__ == "__main__":
    main()
