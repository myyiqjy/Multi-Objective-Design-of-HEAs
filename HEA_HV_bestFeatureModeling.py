import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

    best_features =['MagpieData avg_dev CovalentRadius', 'MagpieData mean Electronegativity',
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


    # 模型构建
    print('\n', '\033[1m随机森林模型构建'.center(120))
    RF = RandomForestRegressor(random_state=100)
    Evaluate(RF, Train_X_std, Train_Y)

    print('\n', '\033[1m多元线性回归模型构建'.center(120))
    MLR = LinearRegression()
    Evaluate(MLR, Train_X_std, Train_Y)

    # Creating a Ridge Regression model
    print('\n', '\033[1m岭回归模型构建'.center(120))
    RLR = Ridge()
    Evaluate(RLR, Train_X_std, Train_Y)

    # Creating a Lasso Regression model
    print('\n', '\033[1mLASSO回归模型构建'.center(120))
    LLR = Lasso()
    Evaluate(LLR, Train_X_std, Train_Y)

    print('\n', '\033[1mLightGBM回归模型构建'.center(120))
    lgbm = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
                             metric='rmse', random_state=100)
    Evaluate(lgbm, Train_X_std, Train_Y)

    print('\n', '\033[1mSVR模型构建'.center(120))

    SVR1 = SVR(kernel='rbf',C=100)
    print("C:",100)
    Evaluate(SVR1, Train_X_std, Train_Y)
    print('\n', '\033[1mBPANN模型构建'.center(120))
    # alpha是正则化参数
    bpann = MLPRegressor(solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(50,20),max_iter=5000, random_state=100)
    Evaluate(bpann, Train_X_std, Train_Y)





if __name__ == "__main__":
    main()
