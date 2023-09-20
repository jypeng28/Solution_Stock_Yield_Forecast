import pandas as pd
from sklearn import linear_model
import pickle


# 读取数据, 选择 'time_id', 'stock_id' 作为索引
train_df = pd.read_csv('../train.csv').set_index(['time_id', 'stock_id'])

# 填充缺失数据
train_df.fillna(0, inplace=True)
print(train_df)

# 创建线性回归模型
reg = linear_model.LinearRegression()

# 将数据集划分为特征和目标变量
X = train_df.iloc[:,:-1].values
y = train_df.iloc[:,-1].values

# 训练模型
reg.fit(X, y)

# 保存模型
with open('reg.pkl', 'wb') as f:
    pickle.dump(reg, f)
