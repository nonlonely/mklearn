"""
使用决策树，使用的特征是 "Pclass","Sex","Age","SibSp","Parch","Fare","Embarked" 其中把空值补上，吧String类型的值转化为int/float类型的值
在这一节，主要是把特征值做处理，主要包含把Pclass，Sex,Age,SibSp,Parch,Fare,Embarked,FamilySize,Title,NameLength
"""
import pandas
def predata(titanic):
    titanic["Age"] = titanic ['Age'] . fillna(titanic['Age'].median())
    titanic["Fare"] = titanic ['Fare'] . fillna(titanic['Fare'].median())
    #Replace all the occurences of male with the number 0.
    titanic.loc[titanic["Sex"] == "male","Sex"] = 0
    titanic.loc[titanic["Sex"] == "female","Sex"] = 1
    titanic["Embarked"] = titanic["Embarked"].fillna('S')     #缺失值用最多的S进行填充
    titanic.loc[titanic["Embarked"] == "S","Embarked"] = 0    #地点用0,1,2
    titanic.loc[titanic["Embarked"] == "C","Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q","Embarked"] = 2
    #Generating a familysize column
    titanic["FamilySize"]=titanic["SibSp"]+titanic["Parch"]
    #The .apply method generates a new series
    titanic["NameLength"]=titanic["Name"].apply(lambda x:len(x))
    import re
    #A function to get the title from a name
    def get_title(name):
        #Use a regular expression to search for a title. Titles always consist of capital and lowercase letters
        title_search = re.search('([A-Za-z]+)\.',name)
        #If the title exists,extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    #Get all the titles and print how often each one occurs.
    titles=titanic["Name"].apply(get_title)
    ##print(pandas.value_counts(titles))
    #Map each titles to an integer.  Some titles are very rare,and are compressed into the same codes as other
    title_mapping = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Dr":5,"Rev":6,"Col":7,"Major":8,"Mlle":9,"Countess":10,"Ms":11,"Lady":12,"Jonkheer":13,"Don":14,"Mme":15,"Capt":16,"Sir":17}
    for k,v in title_mapping.items():
        titles[titles==k]=v
    #Verify that we converted everything.
    ##print(pandas.value_counts(titles))
    #Add in the title column
    titanic["Title"]=titles
    return titanic

titanic=pandas.read_csv("../train.csv")
train=predata(titanic)

import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","Title","NameLength"]

alg=RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=2,min_samples_leaf=1)
#棵决策树，停止的条件：样本个数为2，叶子节点个数为1
#Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
kf=cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
#scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=kf)
train_df = train.filter(regex="Pclass|Sex|Age|SibSp|Parch|Fare|Embarked|FamilySize|Title|NameLength")
print(train_df)
train_np = train_df.as_matrix()
#trainData = train[predictors].filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title|NameLength|FamilySize')
#print(trainData)
#print(train["Survived"].as_matrix())
y=train["Survived"].as_matrix().T
print(y)

alg.fit(train_np,y)
#print(train[predictors])

print("---------")
titanic=pandas.read_csv("../test.csv")
test=predata(titanic)
#print(test.describe())
label=y=test["PassengerId"].as_matrix().T
print(label)
test_df = test.filter(regex="Pclass|Sex|Age|SibSp|Parch|Fare|Embarked|FamilySize|Title|NameLength")
test_np = test_df.as_matrix()

res=alg.predict(test_np).T
print(res)
print(len(label))
print(len(res))
re1=np.vstack((label,res)).T
print(re1)
file_object = open('thefile11.txt', 'w')
for i in range(0, len(re1)):
    file_object.write(re1[i][0]+","+re1[i][1]+"\n")
file_object.close()


