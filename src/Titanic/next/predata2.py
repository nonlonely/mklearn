"""
使用决策树，使用的特征是 "Pclass","Sex","Age","SibSp","Parch","Fare","Embarked" 其中把空值补上，吧String类型的值转化为int/float类型的值
在这一节，主要是把特征值做处理，主要包含把Pclass，Sex,Age,SibSp,Parch,Fare,Embarked,FamilySize,Title,NameLength
"""
import pandas
titanic=pandas.read_csv("../train.csv")

titanic["Age"] = titanic ['Age'] . fillna(titanic['Age'].median())

print(titanic['Sex'].unique())
#Replace all the occurences of male with the number 0.
titanic.loc[titanic["Sex"] == "male","Sex"] = 0
titanic.loc[titanic["Sex"] == "female","Sex"] = 1

titanic["Embarked"] = titanic["Embarked"].fillna('S')     #缺失值用最多的S进行填充
titanic.loc[titanic["Embarked"] == "S","Embarked"] = 0    #地点用0,1,2
titanic.loc[titanic["Embarked"] == "C","Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q","Embarked"] = 2

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

#Initialize our algorithm with the default parameters
#n_estimators is the number of tress we want to make
#min_samples_split is the minimum number of rows we need to m,ake a split
#min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the b)
alg=RandomForestClassifier(random_state=1,n_estimators=100,min_samples_split=2,min_samples_leaf=1)
#棵决策树，停止的条件：样本个数为2，叶子节点个数为1
#Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
kf=cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=kf)

#Take the mean of the scores (because we have one for each fold)
print(scores.mean())


#0.785634118967   n_estimators=10
#0.795735129068   n_estimators=100
