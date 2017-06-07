from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation   #可视化学习的整个过程
import numpy as np
#PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
"""
排名：6263 准确率：0.73684
SibSp 8=7 5=5 4=18 3=16 2=28 共74个 < 100看做一类 记 0，1，2
Parch 2、3、4、5、6共95个也看做一类 记 0，1，2
Age 以[0:16), [16:40),[40:99)离散化处理
Fare 票价与Pclass和Parch、SibSp相关，亲朋通常会入住同一房间
Embarked S计0，C记1，Q记2
Sex 男记1，女记2
"""

x=[]
labelX=[]
f=open('test.csv','r',encoding='UTF-8')
for v  in f:
    arr=v.split(",")
    """
    x.append([int(arr[1]),1 if arr[4]=='male' else 2,1 if arr[5]=="" else 0 if float(arr[5])<16 and float(arr[5])>=0 else 2 if float(arr[5])>=40   else 1,
              float(arr[6]),
              float(arr[7]),0 if arr[9]=="" else float(arr[9]),0 if arr[11]=="S" else 1 if arr[11]=="C" else 2])
    """
    x.append([int(arr[1]),
            1 if arr[4]=='male' else 2,
            1 if arr[5]=="" else 0 if float(arr[5])<18 and float(arr[5])>=0 else 2 if float(arr[5])>=40 else 1,
            2 if int(arr[6])>=2 else int(arr[6]),
            2 if int(arr[7])>=2 else int(arr[7]),
            0 if arr[9]=="" else float(arr[9]),
            0 if arr[11]=="S" else 1 if arr[11]=="C" else 2])
    labelX.append(arr[0])
#print(x)

#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
X=[]
label=[]
f=open('train.csv','r',encoding='UTF-8')
for v  in f:
    arr=v.split(",")
    #一种处理数据特征的方式
    """
    X.append([int(arr[2]),
            1 if arr[5]=='male' else 2,
            1 if arr[6]=="" else 0 if float(arr[6])<16 and float(arr[6])>=0 else 2 if float(arr[6])>=40 else 1,
            float(arr[7]),
            float(arr[8]),
            0 if arr[10]=="" else float(arr[10]),
            0 if arr[12]=="S" else 1 if arr[12]=="C" else 2])
    """
    #另外一种处理数据特征的方式   准确率并没有提高
    X.append([int(arr[2]),
            1 if arr[5]=='male' else 2,
            1 if arr[6]=="" else 0 if float(arr[6])<18 and float(arr[6])>=0 else 2 if float(arr[6])>=40 else 1,
            2 if int(arr[7])>=2 else int(arr[7]),
            2 if int(arr[8])>=2 else int(arr[8]),
            0 if arr[10]=="" else float(arr[10]),
            0 if arr[12]=="S" else 1 if arr[12]=="C" else 2])
    label.append(int(arr[1]))
"""
S=[]
f=open('gender_submission.csv','r',encoding='UTF-8')
for v  in f:
    arr=v.split(",")
    S.append([int(arr[1])])
print(S)


model=RandomForestClassifier()
model.fit(X,label)

print(model.predict(x))   #给训练模型打分，注意用在LinearR中使用R^2 conefficient of determination打分

"""
#交叉验证
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,label, test_size=0.5, random_state=0)
#clf = LogisticRegression()#LR分类 参数默认    #0.793721973094
clf = RandomForestClassifier(n_estimators=8)      #0.807174887892
#clf = DecisionTreeClassifier()  #0.80269058296
clf.fit(X_train, y_train)

re=clf.predict(x)   #给训练模型打分，注意用在LinearR中使用R^2 conefficient of determination打分
print(re)
print(labelX)

re1=np.vstack((labelX,re)).T
print(re1)

file_object = open('thefile.txt', 'w')
for i in range(0, len(re1)):
    file_object.write(re1[i][0]+","+re1[i][1]+"\n")
file_object.close()


"""
y_predicted = clf.predict(X_test).tolist()
print(y_test)
print(y_predicted)
print(clf.score(X_test,y_test))

model=RandomForestClassifier(n_estimators=12, max_depth=3)
model.fit(X,label)

re=model.predict(x)   #给训练模型打分，注意用在LinearR中使用R^2 conefficient of determination打分
print(re)
print(labelX)

re1=np.vstack((labelX,re)).T
print(re1)

file_object = open('thefile.txt', 'w')
for i in range(0, len(re1)):
    file_object.write(re1[i][0]+","+re1[i][1]+"\n")
file_object.close()
#np.savetxt('new.csv',rel,delimiter = ',')
"""