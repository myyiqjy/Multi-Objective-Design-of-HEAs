import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.feature_selection import VarianceThreshold
from CorrDeleteFeature import corrDeleteFeature
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score as CVS
import seaborn as sns

# 输出显示不限制长度
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# 定义一个评价函数
def Evaluate(model, X, Y,Train_cv10_r2,Train_cv10_rmse):
    model_r2 = CVS(model, X, Y, scoring='r2', cv=10)
    model_mse = CVS(model, X, Y, scoring='neg_mean_squared_error', cv=10)
    model_r2_mean = model_r2.mean()
    model_mse_mean = model_mse.mean()
    model_rmse_mean = np.sqrt(-1 * model_mse_mean)
    Train_cv10_r2.append(model_r2_mean)
    Train_cv10_rmse.append(model_rmse_mean)

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

"""
# 绘制Elongation的直方图并添加数据标签
sns.set_style("whitegrid")
# 把'#1a6fdf'设置为默认设置，换成统一的绘图风格
n, bins, patches = plt.hist(x=df["Elongation"],bins=16)
for i in range(len(n)):
    plt.text(bins[i]+0.5*(bins[1]-bins[0]), n[i]*1.02, int(n[i]), fontsize=12, horizontalalignment="center") #打标签，在合适的位置标注每个直方图上面样本数

plt.xlabel('D (%)') #绘制x轴
plt.ylabel('Count') #绘制y轴
plt.ylim(0,35)
plt.savefig('F:\python39_HEA_new\picture\D_count_2.png', dpi=500)
plt.show()
"""

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


# 先用所有特征建模
# 一次建模效果
Train_cv10_r2=[]
Train_cv10_rmse=[]
# print('\n', '\033[1m随机森林模型构建'.center(120))
RF = RandomForestRegressor(random_state=100)
Evaluate(RF, Train_X_std, Train_Y, Train_cv10_r2, Train_cv10_rmse)


# Creating a Ridge Regression model
# print('\n', '\033[1m岭回归模型构建'.center(120))
RLR = Ridge()
Evaluate(RLR, Train_X_std, Train_Y, Train_cv10_r2, Train_cv10_rmse)

# Creating a Lasso Regression model
print('\n', '\033[1mLASSO回归模型构建'.center(120))
LLR = Lasso()
Evaluate(LLR, Train_X_std, Train_Y, Train_cv10_r2, Train_cv10_rmse)

print('\n', '\033[1mLightGBM回归模型构建'.center(120))
lgbm = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
                         metric='rmse', random_state=100)
Evaluate(lgbm, Train_X_std, Train_Y, Train_cv10_r2, Train_cv10_rmse)

print('\n', '\033[1mSVR模型构建'.center(120))
svr100 = SVR(kernel='rbf', C=100)
Evaluate(svr100, Train_X_std, Train_Y, Train_cv10_r2, Train_cv10_rmse)


print("R2------------>",Train_cv10_r2,"\nRMSE-------------->",Train_cv10_rmse)



