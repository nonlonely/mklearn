"""
使用线性回归，使用的特征是 "Pclass","Sex","Age","SibSp","Parch","Fare","Embarked" 其中把空值补上，吧String类型的值转化为int/float类型的值
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

#Import the linear regression class
from sklearn.linear_model import LinearRegression   #线性回归
#Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold    #训练集交叉验证，得到平均值

#The columns we'll use to predict the target
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]     #要输入的特征

#Initialize our algorithm class
alg = LinearRegression()
#Generate cross validation folds for the titanic dataset.   It return the row indices corresponding to train
kf = KFold(titanic.shape[0],n_folds=3,random_state=1)   #样本平均分成3份，交叉验证

predictions = []
for train,test in kf:
    #The predictors we're using to train the algorithm.  Note how we only take then rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    #The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    #Training the algorithm using the predictors and target.
    alg.fit(train_predictors,train_target)
    #We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
print(predictions)
"""
import numpy as np

#The predictions are in three aeparate numpy arrays.    Concatenate them into one.
#We concatenate them on axis 0,as they only have one axis.
predictions = np.concatenate(predictions,axis=0)

#Map predictions to outcomes(only possible outcomes are 1 and 0)
predictions[predictions>.5] = 1
predictions[predictions<=.5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)
"""



























