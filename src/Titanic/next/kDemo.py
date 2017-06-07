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
titanic=predata(titanic)

titanicT=pandas.read_csv("../test.csv")
testT=predata(titanicT)
import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","Title","NameLength"]


#Initialize our algorithm class
alg=RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=2,min_samples_leaf=1)
#Generate cross validation folds for the titanic dataset.   It return the row indices corresponding to train
kf=cross_validation.KFold(titanic.shape[0],n_folds=8,random_state=1)
predictions = []
label=testT['PassengerId'].as_matrix()
print(testT[predictors])


for train,test in kf:
    #The predictors we're using to train the algorithm.  Note how we only take then rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    #The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    #Training the algorithm using the predictors and target.
    alg.fit(train_predictors,train_target)
    #We can now make predictions on the test fold
    test_predictions = alg.predict(testT[predictors].iloc[:,:])
    predictions.append(test_predictions)
#print(predictions[0])
#print(predictions[1])
#print(predictions[2])

first=np.array(predictions[0])
print(first.size)
sec=np.array(predictions[1])
print(label.size)
three=np.array(predictions[2])
import pandas as pd
re=first+sec+three+predictions[3]+predictions[4]+predictions[5]+predictions[6]+predictions[7]
print(re)
s = pd.DataFrame(re)
print(s.size)
s.loc[s[0] == 1,0] = 0
s.loc[s[0] == 2,0] = 0
s.loc[s[0] > 2,0] = 1
#print(label)
res=s[0].as_matrix()
print(res.size)
rel=np.vstack((label,res)).T
print(rel)

file_object = open('thefile.txt', 'w')
for i in range(0, len(rel)):
    str1=str(rel[i][0])+","+str(rel[i][1])
    file_object.write(str1+"\n")
file_object.close()


#当结果是7的时候去大于1的值，这是准确率取得最大值









