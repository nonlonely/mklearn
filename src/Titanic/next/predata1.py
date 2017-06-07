"""
使用逻辑回归，使用的特征是 "Pclass","Sex","Age","SibSp","Parch","Fare","Embarked" 其中把空值补上，吧String类型的值转化为int/float类型的值
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

predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]     #要输入的特征

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression   #逻辑回归
#Initialize our algorithm
alg=LogisticRegression(random_state=1)
#Compute the accuracy score for all the cross validation folds.(much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=3)
#Take the mean of the scores (because we have one for each fold)
print(scores.mean())


#结果0.787878787879