# -*- coding: UTF-8 -*-
#准确率是0.78947
import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

def substrings_in_string1(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    return np.nan

def replace_titles(x):
    title=x['Title']
    if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Master']:
        return 'Master'
    elif title in ['Countess', 'Mme','Mrs']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms','Miss']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    elif title =='':
        if x['Sex']=='Male':
            return 'Master'
        else:
            return 'Miss'
    else:
        return title

#title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
            ##  'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
            ## 'Don', 'Jonkheer']  if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']: Mr

title_list = {'Mrs':'Mrs', 'Mr':'Mr', 'Master':'Master', 'Miss':'Miss', 'Major':'Mr', 'Rev':'Mr',
                'Dr':'Dr', 'Ms':'Miss', 'Mlle':'Miss','Col':'Mr', 'Capt':'Mr', 'Mme':'Mrs', 'Countess':'Mrs',
                'Don':'Mr', 'Jonkheer':'Mr',"Lady":'Miss',"Sir":"Master"}


label = train['Survived'] # 目标列

#除此之外Name、Ticket、Cabin也是离散特征，我们暂时不用这几个特征，直观上来讲，叫什么名字跟在事故中是否存活好像没有太大的联系。
train.groupby(['Pclass'])['PassengerId'].count().plot(kind='bar')
train.groupby(['SibSp'])['PassengerId'].count().plot(kind='bar')

#连续特征处理
##Age、Fare是连续特征，观察数据分布查看是否有缺失值和异常值，我们看到Age中存在缺失值，我们考虑使用均值来填充缺失值。
print('检测训练集是否有缺失值：')
print(train[train['Age'].isnull()]['Age'].head())
print(train[train['Fare'].isnull()]['Fare'].head())
print(train[train['SibSp'].isnull()]['SibSp'].head())
print(train[train['Parch'].isnull()]['Parch'].head())
train['Age'] = train['Age'].fillna(train['Age'].mean())
print('填充之后再检测：')
print(train[train['Age'].isnull()]['Age'].head())
print(train[train['Fare'].isnull()]['Fare'].head())

print('检测测试集是否有缺失值：')
print(test[test['Age'].isnull()]['Age'].head())
print(test[test['Fare'].isnull()]['Fare'].head())
print(test[test['SibSp'].isnull()]['SibSp'].head())
print(test[test['Parch'].isnull()]['Parch'].head())
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
print('填充之后再检测：')
print(test[test['Age'].isnull()]['Age'].head())
print(test[test['Fare'].isnull()]['Fare'].head())

# 处理Title特征
import re
def get_title(name):
    #Use a regular expression to search for a title. Titles always consist of capital and lowercase letters
    title_search = re.search('([A-Za-z]+)\.',name)
    #If the title exists,extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
Ttitles=train["Name"].apply(get_title)
TestTitle=test["Name"].apply(get_title)


#train['Title'] = train['Name'].map(lambda x: substrings_in_string1(x, title_list))
#test['Title'] = test['Name'].map(lambda x: substrings_in_string1(x, title_list))

#train['Title'] = train.apply(replace_titles, axis=1)
#test['Title'] = test.apply(replace_titles, axis=1)

for k,v in title_list.items():
    Ttitles[Ttitles==k]=v
    TestTitle[TestTitle==k]=v

#Verify that we converted everything.
##print(pandas.value_counts(titles))
#Add in the title column
train["Title"]=Ttitles
test["Title"]=TestTitle

# family特征
train['Family_Size'] = train['SibSp'] + train['Parch']
train['Family'] = train['SibSp'] * train['Parch']
test['Family_Size'] = test['SibSp'] + test['Parch']
test['Family'] = test['SibSp'] * test['Parch']

train['AgeFill'] = train['Age']
mean_ages = np.zeros(4)
mean_ages[0] = np.average(train[train['Title'] == 'Miss']['Age'].dropna())
mean_ages[1] = np.average(train[train['Title'] == 'Mrs']['Age'].dropna())
mean_ages[2] = np.average(train[train['Title'] == 'Mr']['Age'].dropna())
mean_ages[3] = np.average(train[train['Title'] == 'Master']['Age'].dropna())
train.loc[ (train.Age.isnull()) & (train.Title == 'Miss') ,'AgeFill'] = mean_ages[0]
train.loc[ (train.Age.isnull()) & (train.Title == 'Mrs') ,'AgeFill'] = mean_ages[1]
train.loc[ (train.Age.isnull()) & (train.Title == 'Mr') ,'AgeFill'] = mean_ages[2]
train.loc[ (train.Age.isnull()) & (train.Title == 'Master') ,'AgeFill'] = mean_ages[3]

train['AgeCat'] = train['AgeFill']
train.loc[ (train.AgeFill<=10), 'AgeCat'] = 'child'
train.loc[ (train.AgeFill>60), 'AgeCat'] = 'aged'
train.loc[ (train.AgeFill>10) & (train.AgeFill <=30) ,'AgeCat'] = 'adult'
train.loc[ (train.AgeFill>30) & (train.AgeFill <=60) ,'AgeCat'] = 'senior'

train['Fare_Per_Person'] = train['Fare'] / (train['Family_Size'] + 1)

test['AgeFill'] = test['Age']
mean_ages = np.zeros(4)
mean_ages[0] = np.average(test[test['Title'] == 'Miss']['Age'].dropna())
mean_ages[1] = np.average(test[test['Title'] == 'Mrs']['Age'].dropna())
mean_ages[2] = np.average(test[test['Title'] == 'Mr']['Age'].dropna())
mean_ages[3] = np.average(test[test['Title'] == 'Master']['Age'].dropna())
test.loc[ (test.Age.isnull()) & (test.Title == 'Miss') ,'AgeFill'] = mean_ages[0]
test.loc[ (test.Age.isnull()) & (test.Title == 'Mrs') ,'AgeFill'] = mean_ages[1]
test.loc[ (test.Age.isnull()) & (test.Title == 'Mr') ,'AgeFill'] = mean_ages[2]
test.loc[ (test.Age.isnull()) & (test.Title == 'Master') ,'AgeFill'] = mean_ages[3]

test['AgeCat'] = test['AgeFill']
test.loc[ (test.AgeFill<=10), 'AgeCat'] = 'child'
test.loc[ (test.AgeFill>60), 'AgeCat'] = 'aged'
test.loc[ (test.AgeFill>10) & (test.AgeFill <=30) ,'AgeCat'] = 'adult'
test.loc[ (test.AgeFill>30) & (test.AgeFill <=60) ,'AgeCat'] = 'senior'

test['Fare_Per_Person'] = test['Fare'] / (test['Family_Size'] + 1)

train.Embarked = train.Embarked.fillna('S')
test.Embarked = test.Embarked.fillna('S')

train.loc[ train.Cabin.isnull() == True, 'Cabin'] = 0.2
train.loc[ train.Cabin.isnull() == False, 'Cabin'] = 1

test.loc[ test.Cabin.isnull() == True, 'Cabin'] = 0.2
test.loc[ test.Cabin.isnull() == False, 'Cabin'] = 1

#Age times class
train['AgeClass'] = train['AgeFill'] * train['Pclass']
train['ClassFare'] = train['Pclass'] * train['Fare_Per_Person']

train['HighLow'] = train['Pclass']
train.loc[ (train.Fare_Per_Person < 8) ,'HighLow'] = 'Low'
train.loc[ (train.Fare_Per_Person >= 8) ,'HighLow'] = 'High'

#Age times class
test['AgeClass'] = test['AgeFill'] * test['Pclass']
test['ClassFare'] = test['Pclass'] * test['Fare_Per_Person']

test['HighLow'] = test['Pclass']
test.loc[ (test.Fare_Per_Person < 8) ,'HighLow'] = 'Low'
test.loc[ (test.Fare_Per_Person >= 8) ,'HighLow'] = 'High'

#print(train.head(1))
# print test.head()

# 处理训练集
Pclass = pd.get_dummies(train.Pclass)
Sex = pd.get_dummies(train.Sex)
Embarked = pd.get_dummies(train.Embarked)
Title = pd.get_dummies(train.Title)
AgeCat = pd.get_dummies(train.AgeCat)
HighLow = pd.get_dummies(train.HighLow)
train_data = pd.concat([Pclass, Sex, Embarked, Title, AgeCat, HighLow], axis=1)
train_data['Age'] = train['Age']
train_data['Fare'] = train['Fare']
train_data['SibSp'] = train['SibSp']
train_data['Parch'] = train['Parch']
train_data['Family_Size'] = train['Family_Size']
train_data['Family'] = train['Family']
train_data['AgeFill'] = train['AgeFill']
train_data['Fare_Per_Person'] = train['Fare_Per_Person']
train_data['Cabin'] = train['Cabin']
train_data['AgeClass'] = train['AgeClass']
train_data['ClassFare'] = train['ClassFare']

cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Family_Size', 'Family', 'AgeFill', 'Fare_Per_Person', 'AgeClass', 'ClassFare']
train_data[cols] = train_data[cols].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
print(train_data.head())
train_data.to_csv( "train_data.csv", index=False )
# 处理测试集
Pclass = pd.get_dummies(test.Pclass)
Sex = pd.get_dummies(test.Sex)
Embarked = pd.get_dummies(test.Embarked)
Title = pd.get_dummies(test.Title)
AgeCat = pd.get_dummies(test.AgeCat)
HighLow = pd.get_dummies(test.HighLow)
test_data = pd.concat([Pclass, Sex, Embarked, Title, AgeCat, HighLow], axis=1)
test_data['Age'] = test['Age']
test_data['Fare'] = test['Fare']
test_data['SibSp'] = test['SibSp']
test_data['Parch'] = test['Parch']
test_data['Family_Size'] = test['Family_Size']
test_data['Family'] = test['Family']
test_data['AgeFill'] = test['AgeFill']
test_data['Fare_Per_Person'] = test['Fare_Per_Person']
test_data['Cabin'] = test['Cabin']
test_data['AgeClass'] = test['AgeClass']
test_data['ClassFare'] = test['ClassFare']

test_data[cols] = test_data[cols].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
print(test_data.head())
test_data.to_csv( "test_data.csv", index=False )
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import RandomForestClassifier

import numpy as np
model_lr = LR(penalty = 'l2', dual = True, random_state = 0)
model_lr.fit(train_data, label)
print("逻辑回归10折交叉验证得分: ", np.mean(cross_val_score(model_lr, train_data, label, cv=10, scoring='roc_auc')))

result = model_lr.predict(test_data)
output = pd.DataFrame( data={"PassengerId":test["PassengerId"], "Survived":result} )
output.to_csv( "lr.csv", index=False, quoting=3 )