import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.feature_selection import VarianceThreshold
from CorrDeleteFeature import corrDeleteFeature
import numpy as np
from sklearn.svm import SVR
from matplotlib.pyplot import MultipleLocator
# 注意这里用的包是sklearn-genetic，不是genetic_selection
from genetic_selection import GeneticSelectionCV


# 输出显示不限制长度
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)


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
    df1.index=df1["formula"].values
    df = df1.drop(["formula","composition_obj", "Weight Fraction", "Atomic Fraction"], axis=1)
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
    Train_X_std = pd.DataFrame(Train_X_std, columns=X.columns,index=Train_X.index)
    # print(Train_X_std.describe())

    print('\n', '\033[1mStandardardization on Testing set'.center(120))
    Test_X_std = std.transform(Test_X)
    Test_X_std = pd.DataFrame(Test_X_std, columns=X.columns,index=Test_X.index)



    # 阈值取0
    def variance_threshold_selector(data, threshold=0):
        selector = VarianceThreshold(threshold)
        selector.fit(data)
        return data[data.columns[selector.get_support(indices=True)]]


    FeatureNum = len(Train_X_std.columns)
    Train_X_std = variance_threshold_selector(Train_X_std)
    VFeatureNum = len(Train_X_std.columns)
    features = Train_X_std.columns
    Test_X_std = Test_X_std[features]
    # VarianceThreshold deleted feature num  --->  16
    print('Original feature num  ---> ', FeatureNum, '\nVarianceThreshold deleted feature num  ---> ',
          FeatureNum - VFeatureNum)

    print('\n', '\033[1m相关系数特征选择'.center(120))
    # 先把特征和目标变量拼成一个dataframe
    data_Train_X_Y = pd.concat([Train_Y, Train_X_std], axis=1)
    # 使用我在CorrDeleteFeature.py编写的函数实现相关系数筛选特征，['deletedFeatures:', ['A', 'B']]，返回的特征名索引是1
    deletedFeatureName = corrDeleteFeature(data_Train_X_Y, 0.9)
    # Original feature num  --->  145
    # After PearsonCorrelationThreshold deleting feature num  --->  86
    # 皮尔逊相关系数法删除了59个特征
    print('Original feature num  ---> ', VFeatureNum, '\nAfter PearsonCorrelationThreshold deleting feature num  ---> ',
          VFeatureNum - len(deletedFeatureName[1]))
    features = [x for x in features if x not in deletedFeatureName[1]]
    Train_X_std = Train_X_std[features]
    Test_X_std = Test_X_std[features]



    # 包装法特征筛选，得到最佳特征子集的数量为28
    print('\n','\033[1m结合svr模型特征筛选'.center(120))
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from time import time
    
    svr = SVR(kernel='rbf',C=100)
    select_feature_all_times=[]
    svr_avg_mse=[]
    tic_fwd = time()
    sfs_forward = SFS(svr, k_features=len(features), forward=True,
                    cv=10,scoring='neg_mean_squared_error')
    sfs_forward.fit(Train_X_std, Train_Y)
    sfs_results=sfs_forward.subsets_
    for i in range(len(features)):
        select_feature_all_times.append(sfs_results[i+1]['feature_names'])
        svr_avg_mse.append(sfs_results[i+1]['avg_score'])
    toc_fwd = time()
    svr_avg_mse=[-1*i for i in svr_avg_mse]
    svr_avg_rmse=[np.sqrt(i) for i in svr_avg_mse]
    print(f"Done in {toc_fwd - tic_fwd:.3f}s")
    #print('SFS RMSE  ---> ',svr_avg_rmse,'\n特征名  ---> ',select_feature_all_times)
    # 画出不同特征数量对应的分数图
    x=range(1,len(features)+1)
    # alpha 0表示完全透明，1表示完全不透明
    plt.plot(x, svr_avg_rmse, linestyle='-.', marker='*', color='#1a6fdf', markerfacecolor='#f14040', markeredgecolor='#f14040',
             alpha=0.5, linewidth=1, ms=10)
    #plt.legend() # 让图例生效
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(svr_avg_rmse)-0.5,max(svr_avg_rmse)+0.5)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"Number of features") #X轴标签
    plt.ylabel("RMSE") #Y轴标签
    #plt.title("一个简单的折线图") #标题
    min_indx=np.argmin(svr_avg_rmse)#min value index
    # 打印最佳特征子集
    best_features=select_feature_all_times[min_indx]
    print('SFS svr 最佳特征子集-------> ',best_features,'\n最佳特征子集数量-------> ',min_indx+1)
    # 标记最低点
    plt.plot(x[min_indx],svr_avg_rmse[min_indx],marker='*', markerfacecolor='g', markeredgecolor='g',
             alpha=0.5,ms=10)
    show_min='['+str(x[min_indx])+','+str(round(svr_avg_rmse[min_indx],2))+']'
    plt.annotate(show_min,xytext=(x[min_indx],svr_avg_rmse[min_indx]+0.2),xy=(x[min_indx],svr_avg_rmse[min_indx]))
    plt.savefig('F:\python39_HEA_new\picture\D_SFS.png', dpi=500)
    plt.show()



    # 包装法特征筛选
    print('\n', '\033[1m遗传算法结合SVR模型进行特征筛选'.center(120))
    svr = SVR(kernel='rbf', C=100)


    featureNum = []
    genetic_RMSE = []
    select_feature_all_times = []
    # 第一次设置为20,50次
    # 第二次设置为15,10次
    # 第三次设置为14,5次
    # 第四次设置为13,5次
    # 第五次设置为12,10次
    # 第六次设置为10,5次
    # 第七次设置为9,5次
    for i in range(5):
        selector = GeneticSelectionCV(svr,
                                      cv=10,
                                      verbose=0,
                                      scoring="neg_mean_squared_error",
                                      max_features=9,
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
    min_feature_indx = np.argmin(featureNum)
    # 打印最佳特征子集
    best_features = select_feature_all_times[min_indx]
    min_features = select_feature_all_times[min_feature_indx]

    try:
        # 画出空心圆，把c设置为空，通过edgecolors设置颜色
        plt.scatter(featureNum, genetic_RMSE, c='none', marker="o", edgecolors='g', linewidths=0.5, s=50)
        # plt.legend() # 让图例生效
        plt.xlim(min(featureNum) - 1, max(featureNum) + 1)
        plt.ylim(min(genetic_RMSE) - 3, max(genetic_RMSE) + 2.5)
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel(u"Number of features")  # X轴标签
        plt.ylabel("RMSE")  # Y轴标签
        # plt.title("一个简单的折线图") #标题
        # 设置x轴刻度为1
        x_major_locator = MultipleLocator(1)
        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 标记每一列的最低点
        featureNum_unique = list(set(featureNum))
        for i in range(len(featureNum_unique)):
            # 列表不能使用布尔值切片
            mask_code = np.array([featureNum[j] == featureNum_unique[i] for j in range(len(featureNum))])
            genetic_RMSE = np.array(genetic_RMSE)
            min_RMSE = min(genetic_RMSE[mask_code])
            genetic_RMSE = list(genetic_RMSE)
            each_num_min_rmse_index = genetic_RMSE.index(min_RMSE)
            # 标记最低点
            plt.scatter(featureNum_unique[i], genetic_RMSE[each_num_min_rmse_index], c="none", edgecolors='darkorange', linewidths=1.5, marker="o", s=50)
            show_min = str(round(genetic_RMSE[each_num_min_rmse_index], 2))
            if i % 2 == 0:
                plt.annotate(show_min, xytext=(featureNum_unique[i] - 0.3, genetic_RMSE[each_num_min_rmse_index] - 0.4),
                             xy=(featureNum_unique[i], genetic_RMSE[each_num_min_rmse_index]))
            else:
                plt.annotate(show_min, xytext=(featureNum_unique[i] - 0.3, genetic_RMSE[each_num_min_rmse_index] - 1),
                             xy=(featureNum_unique[i], genetic_RMSE[each_num_min_rmse_index]))

            print("特征数为%s时最小的RMSE对应的特征" % featureNum_unique[i], select_feature_all_times[each_num_min_rmse_index])
        plt.show()

    except TypeError as reason:
        print('画图出错了T_T')
        print('出错原因是%s' % str(reason))
    except OSError as reason:
        print('画图出错了T_T')
        print('出错原因是%s' % str(reason))




    """
    finally:
        plt.scatter(featureNum, genetic_RMSE, c='g', marker="*", alpha=0.5, s=50)
        # plt.legend() # 让图例生效
        plt.xlim(min(featureNum) - 1, max(featureNum) + 1)
        plt.ylim(min(genetic_RMSE) - 2, max(genetic_RMSE) + 2)
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel(u"Number of features")  # X轴标签
        plt.ylabel("RMSE")  # Y轴标签
        plt.show()
    """
if __name__ == "__main__":
    main()
