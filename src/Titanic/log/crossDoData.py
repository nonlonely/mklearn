from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation   #可视化学习的整个过程

#利用随机森林把年龄确实的值补齐
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df=df[['Age','Fare','Parch','SibSp','Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age=age_df[age_df.Age.notnull()].as_matrix()
    unknown_age=age_df[age_df.Age.isnull()].as_matrix()
    # y即目标年龄
    y=known_age[:,0]
    # X即特征属性值
    X=known_age[:,1:]
    #fit到到RandomForestRegressor之中
    rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)
    # 用得到的模型进行未知年龄结果预测
    predictedAges=rfr.predict(unknown_age[:,1::])
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()),'Age']=predictedAges

    return df,rfr

#把Cabin缺失的值补齐
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin']="Yes"
    df.loc[(df.Cabin.isnull()),'Cabin']="No"
    return df

data_train=pd.read_csv("../train.csv")
print(data_train.head())
print("-----------------")
data_train,rfr=set_missing_ages(data_train)
data_train=set_Cabin_type(data_train)
print(data_train.head())

dummies_Cabin=pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked=pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex=pd.get_dummies(data_train['Sex'],prefix='Sex')
dummies_Pclass=pd.get_dummies(data_train['Pclass'],prefix='Pclass')

df=pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)
#print(df)

#年龄和fare规范化
import sklearn.preprocessing as preprocessing
scaler=preprocessing.StandardScaler()
age_scala_param=scaler.fit(df['Age'])
df['Age_scaled']=scaler.fit_transform(df['Age'],age_scala_param)
fare_scala_param=scaler.fit(df['Fare'])
df['Fare_scaled']=scaler.fit_transform(df['Fare'],fare_scala_param)

#建模过程
from sklearn import linear_model

# 分割数据，按照 训练数据:cv数据 = 7:3的比例
split_train, split_cv = cross_validation.train_test_split(df, test_size=0.3, random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# 生成模型
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])

# 对cross validation数据进行预测

cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(cv_df.as_matrix()[:,1:])

origin_data_train = pd.read_csv("../train.csv")
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
print(bad_cases.head())
"""
#经过上面的过程已经把模型训练出来，接下来需要的是处理测试数据集
data_test=pd.read_csv("../test.csv")
data_test.loc[(data_test.Fare.isnull()),'Fare']=0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df=data_test[['Age','Fare','Parch','SibSp','Pclass']]
null_age=tmp_df[data_test.Age.isnull()].as_matrix()

# 根据特征属性X预测年龄并补上
X=null_age[:,1:]
predictedAges=rfr.predict(X)
data_test.loc[(data_test.Age.isnull()),'Age']=predictedAges

data_test=set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scala_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scala_param)

print(df_test.head())

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
#result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
#result.to_csv("logistic_regression_predictions3.csv", index=False)

"""













